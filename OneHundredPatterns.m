%==========================================================================
% Project 6 (100 Patterns)
%==========================================================================

%% Load
training_images = loadMNISTImages('train-images.idx3-ubyte');
% train_images(:,i) is a double matrix of size 784xi(where i = 1 to 60000)
% Intensity rescale to [0,1]

training_labels = loadMNISTLabels('train-labels.idx1-ubyte');
% train_labels(i) - 60000x1 vector

% Prepare experinment data
number_of_training_images = 100;

[balanced_train_image, balanced_train_labels] = balance_MNIST_selection(...
    training_images,training_labels,number_of_training_images);

T1_train = zeros(784, number_of_training_images);
T1_train_label = zeros(number_of_training_images,1);

for i = 1: number_of_training_images
    T1_train(:,i) = double(imbinarize(balanced_train_image(:,i),0.5));
    T1_train_label(i) = balanced_train_labels(i);
end


% %% Orthoiamge % Uncomment when needed
% Inew = orthoimages();
% T1_train = zeros(784,10);
% for i = 1: 10
%     T1_train(:,i) = reshape(Inew(:,:,i),[],1);
% end

% Initial weights
a = T1_train;
a(a>0.5)=1;
a(a~=1)= -1;

b = a * a';

% Set w_ii to 0;
w = b - diag(diag(b));

% Noise
noise = [0.05, 0.1, 0.15, 0.2]; % [0.3, 0.4, 0.9, 1] Uncomment and replace if needed

for h = 1 : length(noise)
    % Adding salt-and-pepper noise
    I = imnoise(T1_train,'salt & pepper' ,noise(h));
    I(I>0.5)=1;
    I(I~=1)= -1;
   
    % Plot images from number 0 to 9
    rows = 10; % Images rows (Change to 10 next)
    columns = 10; % Images columns (Change to 10 next)
    image = zeros(58,145); % 29*2=58 for rows; 29*5=145 for columns
    task_1_image= ones(29,29);
    subplot(4,3,3*h-2);
    for i = 1:rows
        for j = 1:columns
            n = j + (i-1)*columns;
            task_1_image(1:28,1:28) =reshape(I(:,n),[28,28]);
            
            image(29*(i-1)+1:29*i,29*(j-1)+1:29*j) =task_1_image ;
            
        end
    end
    imshow(image);
    str_train = sprintf('The input image for noise =  %g ',noise(h));
    title(str_train);
    
%% HNN
    error_bar = ones(1,10);
    x_out = zeros(784,100);
    
    for i = 1: 100


        x = I(:,i);
        x = orth(x);
        E = 1;
        E_new = 0;
        count = 1;
        iteration= 0;
        change = 1;
        fprintf("Processing number: %d ", i);
                fprintf('\n');
        while(change~=0)

            Net =  w * x ;
            Net(Net > 0) = 1;
            Net(Net==0)= x(Net==0);
            Net(Net < 0) = -1;
            
            E = - 0.5 * Net' * w * Net;
            change = E - E_new;
            E_new = E;
            x = Net;

            iteration = iteration +1;
            fprintf(1, repmat('\b',1,count));
            count=fprintf("Current iterations: %d, energy: %f ", iteration, E);
            if iteration > 100
                break
            end
        end
        fprintf('\n')
        x_out(:,i) = Net;
        error = sum(abs(x_out(:,i) - T1_train(:,i)))/784;
        error_bar(1,i) = error;
        

    end
    
    % Show the bar chart of the percentage error
    task_1_test= ones(29,29);
    subplot(4,3,3*h-1);
    for i = 1:rows
        for j = 1:columns
            n = j + (i-1)*columns;
            task_1_test(1:28,1:28) =reshape(x_out(:,n),[28,28]);
            image_t(29*(i-1)+1:29*i,29*(j-1)+1:29*j) =task_1_test ;
            
        end
    end
    imshow(image_t);
    str_test = sprintf('The output image for noise =  %g ',noise(h));
    title(str_test);
    
    subplot(4,3,3*h);
    bar (0:99,error_bar,'b')
    str_error = sprintf('Percentage Error for noise = %g ',noise(h));
    title(str_error);
end
%==========================================================================