datadir = '/Users/daeheonkwon/Archives/physionet.org/files/chbmit/1.0.0/';
contents = dir(datadir);

% Initialize an empty cell array to store folder names
folders = {};

% Iterate through the contents and identify folders
for i = 1:length(contents)
    % Check if the item is a directory and not '.' or '..'
    if contents(i).isdir && ~strcmp(contents(i).name, '.') && ~strcmp(contents(i).name, '..')
        % Add the folder name to the 'folders' cell array
        folders = [folders; contents(i).name];
    end
end

folderpaths = fullfile(datadir, folders);
files = dir(fullfile(char(folderpaths(1)), '*.edf'));

%%
tt = edfread(char(fullfile(folderpaths(1), files(1).name)));
info = edfinfo(char(fullfile(folderpaths(1), files(1).name)));
fs = info.NumSamples/seconds(info.DataRecordDuration);

timeDuration = [2000, 2001];
totalSamples = info.NumSamples(1)*(timeDuration(2) - timeDuration(1));

t = (info.NumSamples(1)*timeDuration(1):info.NumSamples(1)*timeDuration(1) + totalSamples-1)/fs(1);
figure(1);
offset = 100;

for signum = 1:info.NumSignals
    y = zeros(1, totalSamples);
    for recnum = 1 : timeDuration(2) - timeDuration(1)
        y((recnum - 1) * info.NumSamples(signum) + 1 : recnum * info.NumSamples(signum)) = tt.(signum){recnum} + (signum - 1) * offset;
    end
    plot(t, y, 'DisplayName', [info.SignalLabels(signum)]);
    hold on;
end

xlabel('Time (seconds)');
ylabel('Amplitude');
title('Time');
legend('Location', 'eastoutside');

freq = (-totalSamples/2 : totalSamples/2-1)*fs(1) / totalSamples;
figure(2);
for signum = 1:1
    y = zeros(1, totalSamples);
    for recnum = 1 : timeDuration(2) - timeDuration(1)
        y((recnum - 1) * info.NumSamples(signum) + 1 : recnum * info.NumSamples(signum)) = tt.(signum){recnum};
    end
    fft_y = myFT(y);
    fft_y = 20*log10(fft_y);
    plot(freq, abs(fft_y), 'DisplayName', [info.SignalLabels(signum)])
    hold on;
end


xlabel('Frequency (Hz)');
ylabel('Amplitude (dB)');
title('Frequency');
legend('Location', 'eastoutside');

