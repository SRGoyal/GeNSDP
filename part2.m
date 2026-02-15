%%%%%%%%%%%%%%%%%%%%%%%% PART-2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% DL with Oversampled Data %%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%
% Step 5. K-Fold Data Partitioning (k = 10)

kfold=10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
features=SyntheticData;
labels=SyntheticLbl;
rownames={'f1';'f2';'f3';'f4';'f5';'f6';'f7';'f8';'f9';'f10';'f11';'f12';'f13';'f14';'f15';'f16';'f17';'f18';'f19';'f20';'label'};
tbl=table(features(:,1),features(:,2),features(:,3),features(:,4),features(:,5),features(:,6),features(:,7),features(:,8),features(:,9),features(:,10),features(:,11),features(:,12),features(:,13),features(:,14),features(:,15),features(:,16),features(:,17),features(:,18),features(:,19),features(:,20),categorical(labels),'VariableNames',rownames);
labelName="label";
classNames = {'1','2'};



%%%%%%%%%%%%%%%%%
% Step 6. Model Architecture Design


numFeatures = size(tbl,2) - 1;
numClasses = numel(classNames);
classWeights = [0.66 0.33];

layers = [
    featureInputLayer(numFeatures,'Normalization', 'zscore')
    fullyConnectedLayer(51)
    lstmLayer(20)
    reluLayer
    fullyConnectedLayer(51)
    %lstmLayer(20)
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer('Classes',classNames, ...
    'ClassWeights',classWeights)
    ];

%%%%%%%%%%%%%%%%%
% Step 7. Training Configuration
miniBatchSize = 16;
options = trainingOptions('adam', ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false);

numObservations = size(tbl,1)
fold    = cvpartition(labels,'kfold',kfold);
Afold   = zeros(kfold,1); 
result   = zeros(kfold,6); 
confmat = 0;
for i = 1:kfold
  train_idx  = fold.training(i);
  test_idx   = fold.test(i);
  tblTrain = tbl(train_idx,:);
tblTest = tbl(test_idx,:);
YTest = tblTest{:,labelName};

%%%%%%%%%%%%%%%%%
% Step 8. Model Training (within each fold)

net = trainNetwork(tblTrain,labelName,layers,options);


%%%%%%%%%%%%%%%%%
% Step 9. Testing & Evaluation (within each fold)

[YPred,score] = classify(net,tblTest(:,1:end-1),'MiniBatchSize',miniBatchSize);
  con        = confusionmat(YTest,YPred);
  confmat    = confmat + con; 
  %%%%%%%%%%%%%%%%
  YTest1d=double(YTest);
YTest1d(YTest1d==2)=0;
YTest2d=~YTest1d;
YTest1=[YTest1d YTest2d];
t=YTest1';

YPred1d=double(YPred);
YPred1d(YPred1d==2)=0;
YPred2d=~YPred1d;
YPred1=[YPred1d YPred2d];
outputs=YPred1';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[c,c_ann,ind,per] = confusion(t,outputs);
  
   TN_ann=c_ann(1,1);
                    FP_ann=c_ann(1,2);
                    FN_ann=c_ann(2,1);
                    TP_ann=c_ann(2,2);
                    P_ann=TP_ann+FN_ann;
                    N_ann=FP_ann+TN_ann;
                    accuracy_ann=(TP_ann+TN_ann)/(P_ann+N_ann)
                    precision_ann=TP_ann/(TP_ann+FP_ann)
                    recall_ann=TP_ann/(TP_ann+FN_ann)
                    Fmeasure_ann=(2*precision_ann*recall_ann)/(precision_ann+recall_ann)
                    mcc_ann=((TP_ann*TN_ann)-(FP_ann*FN_ann))/(sqrt((TP_ann+FP_ann)*(TP_ann+FN_ann)*(TN_ann+FP_ann)*(TN_ann+FN_ann)))
                  

figure(6)
plot(net)

[X_knn,Y_knn,T_knn,AUC_knn,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(YTest,score(:,2),2);
figure,hold off
plot(X_knn,Y_knn)
AUC_knn


    result=[AUC_knn accuracy_ann Fmeasure_ann recall_ann precision_ann mcc_ann]
  Afold(i,1) = sum(diag(con)) / sum(con(:));
end

%%%%%%%%%%%%%%%%%
% Step 10. Cross-Fold Performance Aggregation

Acc  = mean(Afold);
RESULT  = mean(result);




  
figure,hold off
figure,hold off
  
  
  