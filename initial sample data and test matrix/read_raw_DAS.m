%% START
close all; clear all; clc

%%
%files = {"C:\Users\AlbertoMendoza\OneDrive - LYTT\Desktop\test_rawDas\BP Sintef 10m_UTC+0000_DST0_20190620_124000.926.h5"};
files = {"C:\Users\Misael Morales\OneDrive - The University of Texas at Austin\DiReCT Research\Lytt Sintef\BP Sintef 10m_UTC+0000_DST0_20190620_124000.926.h5"};

h5disp(files{1},'/Acquisition')
h5disp(files{1},'/Acquisition/Raw[0]')


%%
time = [];
rawDAS = [];

for i = 1:length(files)
    time = datenum(h5read(files{i},'/Acquisition/Raw[0]/RawDataTime'));
    rawDAS = h5read(files{i},'/Acquisition/Raw[0]/RawData');
end
clear i

%% END