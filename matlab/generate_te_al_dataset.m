function generate_te_al_dataset(input_csv, output_dir, order, n_layers, workers, batch_size, material_csv, max_new_samples)
%GENERATE_TE_AL_DATASET Generate or safely resume a TE/Al RCWA dataset.
%
% max_new_samples is optional and is intended for controlled stop/resume
% tests. Existing seven-argument calls keep their original behaviour.

if nargin < 3 || isempty(order), order = 15; end
if nargin < 4 || isempty(n_layers), n_layers = 50; end
if nargin < 5 || isempty(workers), workers = 4; end
if nargin < 6 || isempty(batch_size), batch_size = 20; end
if nargin < 7 || isempty(material_csv)
    workflow_dir = fileparts(fileparts(mfilename('fullpath')));
    material_csv = fullfile(workflow_dir,'materials','Al_Rakic_BB_400_1700nm.csv');
end
if nargin < 8 || isempty(max_new_samples), max_new_samples = inf; end

if ~exist(input_csv,'file'), error('Input CSV not found: %s',input_csv); end
if ~exist(material_csv,'file'), error('Material table not found: %s',material_csv); end
if ~exist(output_dir,'dir'), mkdir(output_dir); end
if order < 1 || order ~= floor(order), error('order must be a positive integer.'); end
if n_layers < 1 || n_layers ~= floor(n_layers), error('n_layers must be a positive integer.'); end
if workers < 1 || workers ~= floor(workers), error('workers must be a positive integer.'); end
if batch_size < 1 || batch_size ~= floor(batch_size), error('batch_size must be a positive integer.'); end
if max_new_samples < 0 || (~isinf(max_new_samples) && max_new_samples ~= floor(max_new_samples))
    error('max_new_samples must be a non-negative integer or Inf.');
end

inputs = readtable(input_csv);
required_variables = {'LineDensity_per_mm','w1','Theta1_deg','Theta2_deg','Inc_Angle_deg'};
if ~all(ismember(required_variables,inputs.Properties.VariableNames))
    error('Input CSV is missing one or more required columns.');
end
n = height(inputs);
wavelengths = 400:10:1700;
m = length(wavelengths);

paths.spectra = fullfile(output_dir,'te_al_spectra.csv');
paths.r_total = fullfile(output_dir,'te_al_r_total.csv');
paths.passivity = fullfile(output_dir,'te_al_passivity_error.csv');
paths.timing = fullfile(output_dir,'te_al_timing.csv');
paths.inputs = fullfile(output_dir,'te_al_inputs.csv');
paths.wavelengths = fullfile(output_dir,'wavelength_nm.csv');
paths.config = fullfile(output_dir,'run_config.json');
paths.manifest = fullfile(output_dir,'manifest.json');

material_identity = get_material_identity(material_csv);
validate_or_create_provenance(inputs,wavelengths,input_csv,output_dir,paths, ...
    order,n_layers,material_csv,material_identity);

spectra = load_matrix_or_nan(paths.spectra,[n,m]);
r_total = load_matrix_or_nan(paths.r_total,[n,m]);
passivity_error = load_matrix_or_nan(paths.passivity,[n,m]);
sample_elapsed_sec = load_matrix_or_nan(paths.timing,[n,1]);

% The timing file is written last and therefore acts as the commit marker.
% Data in a row without timing are uncommitted and may be safely discarded.
timing_done = isfinite(sample_elapsed_sec);
spectra_done = all(isfinite(spectra),2);
r_total_done = all(isfinite(r_total),2);
passivity_done = all(isfinite(passivity_error),2);
committed_data_complete = spectra_done & r_total_done & passivity_done;
if any(timing_done & ~committed_data_complete)
    bad_rows = find(timing_done & ~committed_data_complete);
    error('Committed checkpoint is incomplete in row(s): %s',mat2str(bad_rows(:)'));
end
uncommitted_rows = ~timing_done & ...
    (any(isfinite(spectra),2) | any(isfinite(r_total),2) | any(isfinite(passivity_error),2));
if any(uncommitted_rows)
    rollback_rows = find(uncommitted_rows);
    spectra(rollback_rows,:) = nan;
    r_total(rollback_rows,:) = nan;
    passivity_error(rollback_rows,:) = nan;
    fprintf('ROLLED_BACK_UNCOMMITTED_ROWS=%s\n',mat2str(rollback_rows(:)'));
end

validate_fully_finite_or_empty(spectra,'te_al_spectra.csv');
validate_fully_finite_or_empty(r_total,'te_al_r_total.csv');
validate_fully_finite_or_empty(passivity_error,'te_al_passivity_error.csv');
if any(isfinite(sample_elapsed_sec) & sample_elapsed_sec < 0)
    error('te_al_timing.csv contains a negative elapsed time.');
end

completed_mask = isfinite(sample_elapsed_sec);
pending_all = find(~completed_mask);
if ~isinf(max_new_samples)
    pending_all = pending_all(1:min(numel(pending_all),max_new_samples));
end

use_parallel = workers > 1 && exist('gcp','file') == 2;
if use_parallel
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= workers
        if ~isempty(pool), delete(pool); end
        parpool('local',workers);
    end
else
    workers = 1;
end
started_all = tic;

for pending_start = 1:batch_size:numel(pending_all)
    pending_end = min(numel(pending_all),pending_start+batch_size-1);
    pending = pending_all(pending_start:pending_end);
    batch_spectra = nan(length(pending),m);
    batch_r_total = nan(length(pending),m);
    batch_passivity = nan(length(pending),m);
    batch_elapsed = nan(length(pending),1);
    if use_parallel
        parfor local_idx = 1:length(pending)
            sample_idx = pending(local_idx);
            [batch_spectra(local_idx,:),batch_r_total(local_idx,:), ...
                batch_passivity(local_idx,:),batch_elapsed(local_idx)] = ...
                solve_one(inputs,sample_idx,wavelengths,order,n_layers,material_csv);
        end
    else
        for local_idx = 1:length(pending)
            sample_idx = pending(local_idx);
            [batch_spectra(local_idx,:),batch_r_total(local_idx,:), ...
                batch_passivity(local_idx,:),batch_elapsed(local_idx)] = ...
                solve_one(inputs,sample_idx,wavelengths,order,n_layers,material_csv);
        end
    end
    if any(~isfinite(batch_spectra),'all') || any(~isfinite(batch_r_total),'all') || ...
            any(~isfinite(batch_passivity),'all') || any(~isfinite(batch_elapsed))
        error('RCWA returned a non-finite value; checkpoint was not committed.');
    end

    spectra(pending,:) = batch_spectra;
    r_total(pending,:) = batch_r_total;
    passivity_error(pending,:) = batch_passivity;

    % Write all physical results first. Timing is the final commit marker.
    write_matrix_atomic(spectra,paths.spectra);
    write_matrix_atomic(r_total,paths.r_total);
    write_matrix_atomic(passivity_error,paths.passivity);
    sample_elapsed_sec(pending) = batch_elapsed;
    write_matrix_atomic(sample_elapsed_sec,paths.timing);

    completed_mask = isfinite(sample_elapsed_sec);
    write_manifest(paths.manifest,input_csv,n,wavelengths,order,n_layers,workers, ...
        batch_size,material_csv,material_identity,toc(started_all), ...
        sample_elapsed_sec,r_total,passivity_error);
    fprintf('CHECKPOINT_COMMITTED=%d/%d\n',sum(completed_mask),n);
end

write_manifest(paths.manifest,input_csv,n,wavelengths,order,n_layers,workers, ...
    batch_size,material_csv,material_identity,toc(started_all), ...
    sample_elapsed_sec,r_total,passivity_error);
fprintf('DATASET_COMPLETED=%d/%d\n',sum(isfinite(sample_elapsed_sec)),n);
fprintf('DATASET_ELAPSED_SEC_CURRENT_PROCESS=%.6f\n',toc(started_all));
end


function validate_or_create_provenance(inputs,wavelengths,input_csv,output_dir,paths,order,n_layers,material_csv,material_identity)
if exist(paths.inputs,'file')
    prior_inputs = readtable(paths.inputs);
    if ~tables_equivalent(prior_inputs,inputs)
        error('Input mismatch: te_al_inputs.csv does not match the requested input CSV.');
    end
else
    write_table_atomic(inputs,paths.inputs);
end

if exist(paths.wavelengths,'file')
    prior_wavelengths = readmatrix(paths.wavelengths);
    if ~isequal(size(prior_wavelengths),size(wavelengths)) || ...
            any(abs(prior_wavelengths(:)-wavelengths(:)) > 1e-12)
        error('Wavelength-grid mismatch with existing checkpoint.');
    end
else
    write_matrix_atomic(wavelengths,paths.wavelengths);
end

config.schema_version = 1;
config.input_csv_at_creation = input_csv;
config.num_samples = height(inputs);
config.wavelength_start_nm = wavelengths(1);
config.wavelength_end_nm = wavelengths(end);
config.wavelength_step_nm = wavelengths(2)-wavelengths(1);
config.order = order;
config.n_layers = n_layers;
config.polarization = 'TE';
config.material_csv_at_creation = material_csv;
config.material_csv_sha256 = material_identity.output_sha256;
config.input_csv_sha256 = sha256_file(input_csv);

if exist(paths.config,'file')
    prior = jsondecode(fileread(paths.config));
    assert_config_equal(prior,config);
    if isfield(prior,'input_csv_sha256')
        if ~strcmp(char(prior.input_csv_sha256),config.input_csv_sha256)
            error('Input CSV hash mismatch. Refusing to mix datasets.');
        end
    else
        % One-time migration for checkpoints created before input hashing.
        prior.input_csv_sha256 = config.input_csv_sha256;
        write_json_atomic(prior,paths.config);
        fprintf('RUN_CONFIG_MIGRATED_WITH_INPUT_HASH=1\n');
    end
else
    validate_legacy_manifest(output_dir,config);
    write_json_atomic(config,paths.config);
end
end


function same = tables_equivalent(left,right)
same = isequal(left.Properties.VariableNames,right.Properties.VariableNames) && ...
    height(left) == height(right) && width(left) == width(right);
if ~same, return; end
for idx = 1:width(left)
    a = left{:,idx};
    b = right{:,idx};
    if isnumeric(a) && isnumeric(b)
        tolerance = 1e-12 .* max(1,max(abs(a),abs(b)));
        values_same = (isnan(a) & isnan(b)) | abs(a-b) <= tolerance;
        if ~all(values_same,'all'), same = false; return; end
    else
        if ~isequal(string(a),string(b)), same = false; return; end
    end
end
end


function assert_config_equal(prior,current)
numeric_fields = {'num_samples','wavelength_start_nm','wavelength_end_nm', ...
    'wavelength_step_nm','order','n_layers'};
for idx = 1:numel(numeric_fields)
    field = numeric_fields{idx};
    if ~isfield(prior,field) || prior.(field) ~= current.(field)
        error('Run configuration mismatch for "%s". Refusing to mix datasets.',field);
    end
end
text_fields = {'polarization','material_csv_sha256'};
for idx = 1:numel(text_fields)
    field = text_fields{idx};
    if ~isfield(prior,field) || ~strcmp(char(prior.(field)),char(current.(field)))
        error('Run configuration mismatch for "%s". Refusing to mix datasets.',field);
    end
end
end


function validate_legacy_manifest(output_dir,config)
legacy_path = fullfile(output_dir,'partial_run_manifest.json');
has_checkpoint = exist(fullfile(output_dir,'te_al_timing.csv'),'file') == 2;
if ~exist(legacy_path,'file')
    if has_checkpoint
        warning('Legacy checkpoint has no run_config.json or partial_run_manifest.json; verify order and n_layers manually.');
    end
    return;
end
legacy = jsondecode(fileread(legacy_path));
if isfield(legacy,'planned_candidates') && legacy.planned_candidates ~= config.num_samples
    error('Legacy manifest sample count does not match the input CSV.');
end
if isfield(legacy,'numerical_resolution')
    resolution = legacy.numerical_resolution;
    if isfield(resolution,'fourier_order') && resolution.fourier_order ~= config.order
        error('Legacy checkpoint order mismatch. Expected %g.',resolution.fourier_order);
    end
    if isfield(resolution,'staircase_layers') && resolution.staircase_layers ~= config.n_layers
        error('Legacy checkpoint n_layers mismatch. Expected %g.',resolution.staircase_layers);
    end
end
if isfield(legacy,'wavelength_nm')
    grid = legacy.wavelength_nm;
    if grid.start ~= config.wavelength_start_nm || grid.stop ~= config.wavelength_end_nm || ...
            grid.step ~= config.wavelength_step_nm
        error('Legacy checkpoint wavelength grid mismatch.');
    end
end
end


function matrix = load_matrix_or_nan(path,expected_size)
matrix = nan(expected_size);
if ~exist(path,'file'), return; end
loaded = readmatrix(path);
if ~isequal(size(loaded),expected_size)
    error('Checkpoint size mismatch for %s: expected %s, got %s.', ...
        path,mat2str(expected_size),mat2str(size(loaded)));
end
matrix = loaded;
end

% 中文代码说明（2026-07-26）：MATLAB 函数 `validate_fully_finite_or_empty`
% 功能：验证 `validate_fully_finite_or_empty` 所对应的实验数据或计算结果。
% 输入：参数：matrix、label
% 输出：无显式返回变量，主要通过文件输出或状态更新完成工作
% 副作用：可能执行数值仿真、输出进度或写入结果；具体以函数体为准
function validate_fully_finite_or_empty(matrix,label)
finite_count = sum(isfinite(matrix),2);
partial_rows = finite_count > 0 & finite_count < size(matrix,2);
if any(partial_rows)
    error('%s contains partially written row(s): %s',label,mat2str(find(partial_rows)'));
end
end


function identity = get_material_identity(material_csv)
identity.output_sha256 = '';
identity.source_table_sha256 = '';
material_manifest_path = strrep(material_csv,'.csv','.manifest.json');
if exist(material_manifest_path,'file')
    material_manifest = jsondecode(fileread(material_manifest_path));
    if isfield(material_manifest,'output_sha256')
        identity.output_sha256 = char(material_manifest.output_sha256);
    end
    if isfield(material_manifest,'source_table_sha256')
        identity.source_table_sha256 = char(material_manifest.source_table_sha256);
    end
end
if isempty(identity.output_sha256)
    error('Material manifest with output_sha256 is required for safe resume: %s',material_manifest_path);
end
actual_sha256 = sha256_file(material_csv);
if ~strcmpi(actual_sha256,identity.output_sha256)
    error('Material CSV hash does not match its manifest: %s',material_csv);
end
end


function hash = sha256_file(path)
md = java.security.MessageDigest.getInstance('SHA-256');
fid = fopen(path,'rb');
if fid < 0, error('Cannot open file for SHA-256: %s',path); end
cleanup = onCleanup(@() fclose_if_open(fid));
while ~feof(fid)
    block = fread(fid,8192,'*uint8');
    if ~isempty(block), md.update(typecast(block,'int8')); end
end
digest = typecast(md.digest(),'uint8');
hash = lower(reshape(dec2hex(digest,2).',1,[]));
fclose(fid);
clear cleanup;
end


function write_manifest(path,input_csv,n,wavelengths,order,n_layers,workers,batch_size,material_csv,material_identity,elapsed,sample_elapsed_sec,r_total,passivity_error)
manifest.updated_local = datestr(now,31);
manifest.status = 'partial';
manifest.input_csv = input_csv;
manifest.num_samples = n;
manifest.wavelength_start_nm = wavelengths(1);
manifest.wavelength_end_nm = wavelengths(end);
manifest.wavelength_step_nm = wavelengths(2)-wavelengths(1);
manifest.order = order;
manifest.n_layers = n_layers;
manifest.workers = workers;
manifest.batch_size = batch_size;
manifest.polarization = 'TE';
manifest.substrate = 'opaque Al';
manifest.material_model = 'Rakic 1998 Brendel-Bormann Al';
manifest.material_model_doi = '10.1364/AO.37.005271';
manifest.material_csv = material_csv;
manifest.material_csv_sha256 = material_identity.output_sha256;
manifest.material_source_table_sha256 = material_identity.source_table_sha256;
manifest.target_harmonic_mapping = 1;
manifest.elapsed_sec_current_process = elapsed;
manifest.completed_samples = sum(isfinite(sample_elapsed_sec));
if manifest.completed_samples == n, manifest.status = 'complete'; end
manifest.max_passivity_error = max(passivity_error,[],'all','omitnan');
manifest.max_total_reflection = max(r_total,[],'all','omitnan');
manifest.mean_sample_elapsed_sec = mean(sample_elapsed_sec,'omitnan');
write_json_atomic(manifest,path);
end


function write_matrix_atomic(data,path)
tmp = [tempname(fileparts(path)) '.csv'];
cleanup = onCleanup(@() delete_if_exists(tmp));
writematrix(data,tmp);
[ok,message] = movefile(tmp,path,'f');
if ~ok, error('Atomic checkpoint replace failed for %s: %s',path,message); end
clear cleanup;
end


function write_table_atomic(data,path)
tmp = [tempname(fileparts(path)) '.csv'];
cleanup = onCleanup(@() delete_if_exists(tmp));
writetable(data,tmp);
[ok,message] = movefile(tmp,path,'f');
if ~ok, error('Atomic table replace failed for %s: %s',path,message); end
clear cleanup;
end


function write_json_atomic(data,path)
tmp = [tempname(fileparts(path)) '.json'];
cleanup = onCleanup(@() delete_if_exists(tmp));
fid = fopen(tmp,'w');
if fid < 0, error('Cannot create temporary JSON file: %s',tmp); end
file_cleanup = onCleanup(@() fclose_if_open(fid));
fprintf(fid,'%s',jsonencode(data));
fclose(fid);
clear file_cleanup;
[ok,message] = movefile(tmp,path,'f');
if ~ok, error('Atomic JSON replace failed for %s: %s',path,message); end
clear cleanup;
end


function delete_if_exists(path)
if exist(path,'file'), delete(path); end
end


function fclose_if_open(fid)
try
    fclose(fid);
catch
end
end


function [spectrum,r_total,passivity,elapsed] = solve_one(inputs,sample_idx,wavelengths,order,n_layers,material_csv)
started_sample = tic;
line_density = inputs.LineDensity_per_mm(sample_idx);
w1 = inputs.w1(sample_idx);
theta1 = inputs.Theta1_deg(sample_idx);
theta2 = inputs.Theta2_deg(sample_idx);
incidence = inputs.Inc_Angle_deg(sample_idx);
period_um = 1000/line_density;
[de1,r1,p1] = RCWA_Engine_Config(period_um,theta1,wavelengths,incidence,order,n_layers,1,material_csv);
if abs(theta1-theta2) < 1e-12
    spectrum = de1;
    r_total = r1;
    passivity = p1;
else
    [de2,r2,p2] = RCWA_Engine_Config(period_um,theta2,wavelengths,incidence,order,n_layers,1,material_csv);
    spectrum = w1*de1+(1-w1)*de2;
    r_total = w1*r1+(1-w1)*r2;
    passivity = w1*p1+(1-w1)*p2;
end
elapsed = toc(started_sample);
end
