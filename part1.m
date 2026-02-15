%%%%%%%%%%%                 SRG            %%%%%%%%%%%%%% 

clear;
close all;
clc;

% Step 1. Data Collection
load ivy1.2.mat;  

%%%%%%%%%%%%%%%%%
% Step 2. Data Preparation

original_data=reshape(meas,1,[]); 
Target(1:312)=1;Target(313:352)=2;Target=Target'; 
real_data_mean = mean (original_data);
real_data_std = std (original_data);

% Step 3. Data Preprocessing
  featureInputLayer(20,'Normalization','zscore');


%%%%%%%%%%%%%%%%%
% Step 4. 1-D GAN-Based Oversampling


generator = @(z) original_data; 
discriminator = @(x) (x - original_data); 


num_samples = 500;
num_epochs = 4;
batch_size = 160;
learning_rate = 0.01;
Runs= 5; 
for i=1:Runs
for epoch = 1:num_epochs
for batch = 1:num_samples/batch_size
% Generate noise samples for the generator
noise = randn(batch_size, 1);
% Generate synthetic data using the generator
synthetic_data = generator(noise);
% Train the discriminator to distinguish real from synthetic data
discriminator_loss = mean((discriminator(synthetic_data) - noise).^2);
% Update the generator to fool the discriminator
generator_loss = mean((discriminator(generator(noise)) - noise).^2);
% Update the generator and discriminator parameters
generator = @(z) generator(z) - learning_rate * generator_loss;
discriminator = @(x) discriminator(x) - learning_rate * discriminator_loss;
end
Run = [' Epoch "',num2str(epoch)];
disp(Run);
end
% Generate synthetic data 
noise_samples = randn(num_samples, 1);
synthetic_data= generator(noise_samples);
Syn(i,:)=synthetic_data;
Run2 = [' Run "',num2str(Runs)];
disp(Run2);
end
S = size(Syn(Runs)); SO = size (meas); SF = SO (1,2); SO = SO (1,1); 
for i=1:Runs
Syn2{i}=reshape(Syn(i,:),[SO,SF]);
Syn2{i}(:,end+1)=Target; 
end
Synthetic3 = cell2mat(Syn2');

%%%%%%%%%%%%%%%%%

%% Merge real + synthetic samples to form balanced dataset.
% Return Oversampled Dataset
SyntheticData=Synthetic3(:,1:end-1); 
SyntheticLbl=Synthetic3(:,end);


%%%%%%%%%%%%%%%%%
