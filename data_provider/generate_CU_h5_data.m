clear; close all; clc;




%-------------------------------------------------------------------------%
%---         path parameters setting-up                                ---%
%-------------------------------------------------------------------------%

% define a root path that contains '/code' and '/data' folder
root_path = '/home/brojackfeely/Research/Video_CTU_withHuQiang';

% read .dat file into work space
data_path = [root_path, '/data/ICME_data/preprocessed_data'];

if (~exist(data_path))
   
    display('!PATH ERROR: No such directory or file...')
    
end

%-------------------------------------------------------------------------%










%-------------------------------------------------------------------------%
%---            load .dat files into worksapce                         ---% 
%-------------------------------------------------------------------------%

% filename example: 
%    e.g. CU64Samples_AI_CPIH_768_1536_2880_4928_qp22_Train.dat
filename = [data_path, ...
            '/CU64Samples_AI_CPIH_768_1536_2880_4928_qp22_Train.dat'];

CU_depth = 64;

offset = CU_depth^2 + 1;

% fopen a file-stream to read/write .dat file...
fid = fopen(filename, 'r');

counter       = 0;      % 'counter': number of blocks i.e. 1 fread -> 1 block
                        %            when 'counter' reaches at 'limit_blocks', it will be set to 0 again
element_count = 0;      % 'element_count': number of elements read in each step 

num_hdf5   = 0;         % 'num_hdf5': number of .hdf5 files 
lim_blocks = 100000;



mode = 'train';
savePath = ['/home/brojackfeely/Research/Video_CTU_withHuQiang/data/ICME_', mode];
batchSize = 10;

if ~exist(savePath, 'dir')   
    dos(['mkdir ', savePath]);
else
    dos(['rm ', savePath, '/*']);
end




data_stack  = zeros(1, CU_depth, CU_depth, lim_blocks);  
label_stack = zeros(1, lim_blocks);

while (~feof(fid))                          % check if reach the end of file stream

    [frame, element_count] = fread(fid, offset, '*uint8');   % '*uint8' equivalent to 'uint8=>uint8'     
    
    if(element_count < CU_depth)
        break;
    end
    
    counter = counter + 1;
    
    % split '/label' and '/data' filed 
    sample_label = frame(1);
    
    sample_data  = reshape(frame(2:end), [CU_depth, CU_depth]);
    sample_data  = sample_data';
    
    data_stack(1, :, :, counter) = sample_data;
    label_stack(counter) = sample_label;
    
    if( counter >= lim_blocks )
        
        filename = ['train_', num2str(num_hdf5), '.h5'];
       
        randOrder = store_Hdf5_file(data_stack, label_stack, savePath, filename, batchSize);
        
        num_hdf5 = num_hdf5 + 1;
        
        counter = 0;
        
        data_stack  = zeros(1, CU_depth, CU_depth, 1, lim_blocks);  
        label_stack = zeros(1, lim_blocks);
        
    end
    
    
    
    % status= fseek(fid, offset, 0); % offset from current position
                                     %     -1 or 'bof': beginning of the file
                                     %      0 or 'cof': current position in file
                                     %      1 or 'eof': end of the file
    
    
end

fclose(fid);
%-------------------------------------------------------------------------%


