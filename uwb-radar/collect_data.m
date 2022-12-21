% Modified from code by Justin Hadella
clear;
clc;

r = vcom_xep_radar_connector('COM6'); % adjust for COM port
r.Open('X4');

dac_min = 949;
dac_max = 1100;
pps = 50;
iterations = 64;


r.TryUpdateChip('pps', pps);
r.TryUpdateChip('dac_min', dac_min);
r.TryUpdateChip('dac_max', dac_max);
r.TryUpdateChip('iterations', iterations);
r.TryUpdateChip('ddc_en', 0);

bins = r.Item('num_samples');

fprintf('bins = %d\n', bins);

prf = r.Item('prf');
fprintf('prf = %d\n', prf);
iterations = r.Item('iterations');
fprintf('iterations = %d\n', iterations);

frameSize = r.numSamplers;   % Get # bins/samplers in a frame
frame = zeros(1, frameSize); % Preallocate frame

name = "back_metal_plate_vertical_empty_box";


save(name + '.mat','frame');
plot(frame);
r.Close();