pathToTrainingData = 'C:/Users/sdick/Google Drive/College/Year 5/5C4 Speech and Audio Engineering/Matlab/TIMIT_database/train/DR5';
pathToTestingData = 'C:/Users/sdick/Google Drive/College/Year 5/5C4 Speech and Audio Engineering/Matlab/TIMIT_database/test/DR5';
pathToResultFiles = 'C:/Users/sdick/OneDrive/Desktop/Results_test';
pathToUnknownData = 'C:/Users/sdick/OneDrive/Desktop/Unknown';

[gmm_v,gmm_uv] = buildGmms(pathToTrainingData,2);

%evaluate the system with test data
results = evaluateMySystem(pathToTestingData,gmm_v,gmm_uv);

fprintf('Rapt Accuracy: %f\n',mean(results(:,1))*100);
fprintf('Rapt Max: %f\n',max(results(:,1))*100);
fprintf('Rapt Min: %f\n',min(results(:,1))*100);
fprintf('My System Accuracy: %f\n',mean(results(:,2))*100);
fprintf('My System Max: %f\n',max(results(:,2))*100);
fprintf('My System Min: %f\n',min(results(:,2))*100);
fprintf('My System voiced precision: %f\n',mean(results(:,3))*100);
fprintf('My System unvoiced precision: %f\n',mean(results(:,4))*100);
fprintf('My System voiced recall: %f\n',mean(results(:,5))*100);
fprintf('My System unvoiced recall: %f\n',mean(results(:,6))*100);
fprintf('RAPT voiced precision: %f\n',mean(results(:,7))*100);
fprintf('RAPT unvoiced precision: %f\n',mean(results(:,8))*100);
fprintf('RAPT voiced recall: %f\n',mean(results(:,9))*100);
fprintf('RAPT unvoiced recall: %f\n',mean(results(:,10))*100);

%test each wav file in test set
function f = evaluateMySystem(pathToTestingData,gmm_v,gmm_uv)
    testingFolders = dir(strcat(pathToTestingData,"/M*"));
    results = [];
    %go through each testing folder
    for i = 1 : length(testingFolders)
        fprintf('Testing Folder #%d of 17 = %s\n', i, testingFolders(i).name);
        %get all wav files in folder
        wavfiles = dir(sprintf(strcat(pathToTestingData,'/%s/*.WAV'),testingFolders(i).name));
        for j = 1 : length(wavfiles)
            cd(sprintf(strcat(pathToTestingData,'/%s'),testingFolders(i).name));
            file = wavfiles(j).name;
            file =file(1:end-4);
            phn = PHN(file);
            rapt = RAPT(file);
            [system_expand,system] = my_system_vuv(file,gmm_v,gmm_uv);
            %output to file
            produceOutputFiles(system,file,strcat(pathToTestingData,'/',testingFolders(i).name));
            
            %get evaluation metrics
            
            [phn_rapt,phn_my_system] = accuracy(system_expand,phn,rapt);
            
            v_prec = voiced_precision(system_expand,phn);
            uv_prec = unvoiced_precision(system_expand,phn);
            v_recall = voiced_recall(system_expand,phn);
            uv_recall = unvoiced_recall(system_expand,phn);
            
            rapt_v_prec = voiced_precision(rapt,phn);
            rapt_uv_prec = unvoiced_precision(rapt,phn);
            rapt_v_recall = voiced_recall(rapt,phn);
            rapt_uv_recall = unvoiced_recall(rapt,phn);            
            
            results = [results;phn_rapt,phn_my_system,v_prec,uv_prec,v_recall,uv_recall,rapt_v_prec,rapt_uv_prec,rapt_v_recall,rapt_uv_recall];
        end
    end
    f = results;
end

%builds the voiced and unvoiced gmms
function [gmm_v,gmm_uv] = buildGmms(pathToTrainingData,mixtures)
    % Get all folders in DR5 train folder that start with M
    trainingFolders = dir(strcat(pathToTrainingData,"/M*"));

    total_voiced = [];
    total_unvoiced = [];

    %get total voiced and total unvoiced from all data in train/DR5
    for i = 1 : length(trainingFolders)
        fprintf('Training Folder #%d of 45 = %s\n', i, trainingFolders(i).name);
        %get all wav files in folder
        wavfiles = dir(sprintf(strcat(pathToTrainingData,'/%s/*.WAV'),trainingFolders(i).name));
        %make sure in the right folder
        cd(sprintf(strcat(pathToTrainingData,'/%s'),trainingFolders(i).name));
        %split each wav file into v/uv and add to total files
        for j = 1 : length(wavfiles)
            %remove .WAV extension
            wavfile = wavfiles(j).name(1:end-4);
            %split into voiced and unvoiced
            [v,uv] = split_vuv(wavfile);
            %add to total v/uv files
            total_voiced = [total_voiced;v];
            total_unvoiced = [total_unvoiced;uv];
        end
    end

    %extract cepstrum features
    fs = 16000; %sampling frequency
    ms10=floor(fs*0.01);
    voiced_features = melcepst(total_voiced,fs,"E0dD",12,floor(3*log(fs)),ms10,ms10,0,0.5);
    unvoiced_features = melcepst(total_unvoiced,fs,"E0dD",12,floor(3*log(fs)),ms10,ms10,0,0.5);

    %build voiced gmm and unvoiced gmm
    gmm_v = fitgmdist(voiced_features,mixtures,'Options',statset('Display','final'));
    gmm_uv = fitgmdist(unvoiced_features,mixtures,'Options',statset('Display','final'));
end

%function splits data into voiced/unvoiced based on phn file
function [voiced,unvoiced] = split_vuv(filename)
    unvoiced_phonetics = {'h#','f','th','s','sh','p','t','k','pau','epi'};
    %add extensions
    wavfile = sprintf('%s.WAV',filename);
    phnfile = sprintf('%s.PHN',filename);
    %read wavfile
    x = readsph(wavfile);
    %read in phn file
    fileID = fopen(phnfile,'r');
    phn = textscan(fileID,'%d %d %s');
    fclose(fileID);
    %convert phonetics to 0(unvoiced) and 1(voiced)
    phn_vuv = [];
    for i = 1:length(phn{1,1})
        if(ismember(phn{1,3}(i),unvoiced_phonetics))
            phn_vuv(i,1) = 0;
        else
            phn_vuv(i,1) = 1;
        end
    end
    %expand phn_vuv to same length as x
    phn_expand = transpose(1:length(x));
    for i = 1:length(phn_expand)
        for j = 1:length(phn_vuv)
            if(phn{1,1}(j) <= phn_expand(i,1) && phn_expand(i,1) <= phn{1,2}(j))
                phn_expand(i,1) = phn_vuv(j);
                break;
            end      
        end
    end
    %split voiced and unvoiced
    voiced = [];
    unvoiced = [];
    for i = 1:length(phn_expand)
        if(phn_expand(i,1) == 1)
            voiced = [voiced;x(i)];
        else
            unvoiced = [unvoiced;x(i)];
        end
    end
end

%compares rapt phn and my system
function [rapt_acc,my_system_acc] = accuracy(my_system,rapt,phn)
    %crop phn and rapt length to same as test_system
    phn = phn(1:(length(my_system)),1:end);
    rapt = rapt(1:(length(my_system)),1:end);
    len = length(my_system);
    %compare phn to rapt
    result = ~xor(phn(:,2),rapt(:,2));
    number_of_1s = sum(result);
    rapt_acc = number_of_1s/len;
    %comapre phn to my_system
    result = ~xor(phn(:,2),my_system(:,2));
    number_of_1s = sum(result);
    my_system_acc = number_of_1s/len;
end

%produces v/uv decision for file using my system(trained gmms)
function [result_expand,result] = my_system_vuv(filename,gmm_v,gmm_uv)
    %add extensions
    wavfile = sprintf('%s.WAV',filename);
    %read in wav file
    [x,fs] = readsph(wavfile);
    ms10 = floor(fs*0.01);
    features = melcepst(x,fs,"E0dD",12,floor(3*log(fs)),ms10,ms10,0,0.5);
    %make decision using posterior prob for each window
    my_decision = [];
    for i = 1:length(features)
        [P_v,nlogL_v] = posterior(gmm_v,features(i,:));
        [P_uv,nlogL_uv] = posterior(gmm_uv,features(i,:));
        if(nlogL_v < nlogL_uv)
        	my_decision(i,1) = 1; %voiced
        else
        	my_decision(i,1) = 0; %unvoiced
        end
    end
    %get sample no.s
    result = [];
    s = 0;
    e = 160;
    for i = 1:length(my_decision)
        result(i,1) = s;
        result(i,2) = e;
        result(i,3) = my_decision(i,1);
        s = e;
        e = e + 160;
    end
    %expand length of file to length of x
    result_expand = transpose(1:(160*length(my_decision)));
    for i = 1:length(result_expand)
        for j = 1:length(result)
            if(result(j,1) <= result_expand(i,1) && result_expand(i,1) <= result(j,2))
                result_expand(i,2) = result(j,3);
                break;
            end     
        end
    end 
end

%function produces v/uv decision for file using timit phonetics
function f = PHN(filename)
    unvoiced_phonetics = {'h#','f','th','s','sh','p','t','k','pau','epi'};
    %add extensions
    phnfile = sprintf('%s.PHN',filename);
    wavfile = sprintf('%s.WAV',filename);
    %read in phnfile
    fileID = fopen(phnfile,'r');
    phn = textscan(fileID,'%d %d %s');
    fclose(fileID);
    %read in wavfile
    x = readsph(wavfile);
    %convert phonetics to 0(unvoiced) and 1(voiced)
    phn_vuv = [];
    for i = 1:length(phn{1,1})
        if(ismember(phn{1,3}(i),unvoiced_phonetics))            
            phn_vuv(i,1) = 0;
        else
            phn_vuv(i,1) = 1;
        end
    end
    %expand phn_vuv to length of x
    phn_expand = transpose(1:length(x));
    for i = 1:length(phn_expand)
        for j = 1:length(phn_vuv)
            if(phn{1,1}(j) <= phn_expand(i,1) && phn_expand(i,1) <= phn{1,2}(j))
                phn_expand(i,2) = phn_vuv(j);
                break;
            end      
        end
    end
    f = phn_expand;
end

%function produces v/uv decision for file using voicebox's fxrapt
function f = RAPT(filename)
    %add extension
    wavfile = sprintf('%s.WAV',filename);
    [x,fs] = readsph(wavfile);
    %use fxrapt function from voicebox
    [vuv,rapt] = fxrapt(x,fs,'u');
    %convert vuv to 0(voiced) and 1(unvoiced)
    for i = 1:length(vuv)
        if(isnan(vuv(i)))
            vuv(i,1) = 0; %unvoiced or silence
        else
            vuv(i,1) = 1; %voiced
        end
    end
    %replace rapt column 3 with vuv column 1
    rapt(:,3) = vuv(:,1);
    %expand rapt to same length as x
    rapt_expand = transpose(1:length(x));
    for i = 1:length(rapt_expand)
        for j = 1:length(rapt)
            if(rapt(j,1) <= rapt_expand(i,1) && rapt_expand(i,1) <= rapt(j,2))
                rapt_expand(i,2) = rapt(j,3);
                break;
            end     
        end
    end
    f = rapt_expand;
end

%produces the output files
function produceOutputFiles(my_system,filename,folderpath)
    if(exist(folderpath,'dir'))
    	cd(folderpath);
    else
        mkdir(folderpath);
        cd(folderpath);
    end
    fileID = fopen(sprintf('%s.VUV',filename),'w');
    fprintf(fileID,'%5d %5d %5d\n', [my_system(:,1),my_system(:,2),my_system(:,3)].');
    fclose(fileID);   
end

%run system on unknown files and produce output files
function runUnknownData(gmm_v,gmm_uv,pathToUnknownData,pathToResultFiles)
	%get all wav files in folder
	wavfiles = dir(strcat(pathToUnknownData,'/*.WAV'));
    for i = 1 : length(wavfiles)
    	file = wavfiles(i).name(1:end-4);
        cd(pathToUnknownData);
    	[system_expand,system] = my_system_vuv(file,gmm_v,gmm_uv);
        rapt = RAPT(file);
    	produceOutputFiles(system,file,pathToResultFiles);
        %[rapt_v_rapt,system_v_rapt] = accuracy(system_expand,rapt,rapt);
        %fprintf('My System v Rapt: %f\n',system_v_rapt);

    end
end

%precision - proportion of positives labeled correctly

%voiced precision
function f = voiced_precision(my_system,phn)
    %crop phn and rapt length to same as test_system
    phn = phn(1:(length(my_system)),1:end);
    
    true_voiced = sum(and(my_system(:,2),phn(:,2)));
    total_voiced = sum(my_system(:,2));
    f = true_voiced/total_voiced;
end

%unvoiced precision
function f = unvoiced_precision(my_system,phn)
    %crop phn and rapt length to same as test_system
    phn = phn(1:(length(my_system)),1:end);
    
    true_unvoiced = sum(~or(my_system(:,2),phn(:,2)));
    total_unvoiced = length(my_system)-sum(my_system(:,2));
    f = true_unvoiced/total_unvoiced;
end

%recall - proportion of actual positives labeled correctly

%voiced recall
function f = voiced_recall(my_system,phn)
    %crop phn and rapt length to same as test_system
    phn = phn(1:(length(my_system)),1:end);
    
    true_voiced = sum(and(my_system(:,2),phn(:,2)));
    total_voiced = sum(phn(:,2));
    f = true_voiced/total_voiced;
end

%unvoiced recall
function f = unvoiced_recall(my_system,phn)
    %crop phn and rapt length to same as test_system
    phn = phn(1:(length(my_system)),1:end);
    
    true_unvoiced = sum(~or(my_system(:,2),phn(:,2)));
    total_unvoiced = length(phn)-sum(phn(:,2));
    f = true_unvoiced/total_unvoiced;
end


