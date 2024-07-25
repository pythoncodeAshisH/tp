import os
import shutil
import pandas as pd
from os import path

from sklearn.cluster import KMeans
import numpy as np

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score

from imblearn.over_sampling import RandomOverSampler,SMOTE
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.model_selection import train_test_split


def update_main_code(process_raw_file_path, failure_log_raw_file_path, n_clusters, max_depth, learning_rate):
    # Call the main functions with the updated parameters
    convert_csv_to_file_struct(process_raw_file_path, failure_log_raw_file_path)
    Output_RUL_calculation_csv = CalculateRUL("SegregatedData/1","DieChangeActionLogSegregation/1")
    RUL_category_file = convert_RUL_to_category("Output_RUL_calculation_csv.csv")
    param = {'max_depth': max_depth, 'learning_rate': learning_rate}
    Train_model("RUL_category_file.csv", n_clusters, param)

# Function to convert csv into file structure
def convert_csv_to_file_struct(process_raw_file_path, failure_log_raw_file_path):
    ProcessData = pd.read_csv(process_raw_file_path)
    
    NameList = ProcessData['Name'].tolist()
    setNameList = list(set(NameList))
    
    setToolList = ['1']
    
    # Main folder name
    main_folder = "SegregatedData"

    # Create the main folder if it doesn't exist, otherwise, clear its contents
    if os.path.exists(main_folder):
        shutil.rmtree(main_folder)
    os.mkdir(main_folder)

    # Create subfolders within the main folder
    for tool in setToolList:
        subfolder_path = os.path.join(main_folder, tool)
        os.mkdir(subfolder_path)
        
        
    for Tool in setToolList:
        print("Tool::::",Tool)
        for Name in setNameList:
            (ProcessData.loc[(ProcessData['Name'] == Name) & (ProcessData['Tool#'] == int(Tool)) ]).to_csv(main_folder + '/' + str(Tool) + '/' + Name + '.csv')
            
            
    LogData = pd.read_csv("ServiceLog.csv")

    NameList = LogData['DriveName'].tolist()
    setNameList = list(set(NameList))

    setToolList = ['1']
    
    # Main folder name
    main_folder = "DieChangeActionLogSegregation"

    # Create the main folder if it doesn't exist, otherwise, clear its contents
    if os.path.exists(main_folder):
        shutil.rmtree(main_folder)
    os.mkdir(main_folder)

    # Create subfolders within the main folder
    for tool in setToolList:
        subfolder_path = os.path.join(main_folder, tool)
        os.mkdir(subfolder_path)
        
        
    for Tool in setToolList:
        print("Tool::::",Tool)
        List = []
        for Name in setNameList:
            df = LogData.loc[(LogData['DriveName'] == Name) & (LogData['Tool'] == int(Tool))]
            df.to_csv('DieChangeActionLogSegregation/' + str(Tool) + '/' + Name + '.csv')
            List.append([Name,list(df.shape)[0]])
        dftoright = pd.DataFrame(List, columns =['Name','CountOfDieChange'])
        dftoright.to_csv(main_folder + '/' + str(Tool) + '/' + 'COUNT.csv')
    
# Function for calculating RUL for each category
def CalculateRUL (process_data_path, failure_data_path):
    
    def GetCycles(ProcessData,ChangeLogData):
        # print(Tool + "::::::::::::::" + ProcessData + "::::::::::::::" + ChangeLogData)
        # print(":::::::::",ProcessData)
        OpsDataRaw = pd.read_csv(ProcessData)
        # print("OpsDataRaw------------->",OpsDataRaw)
        OpsData = OpsDataRaw.sort_values(by=['DatePDString'])
        
        ServiceData = pd.read_csv(ChangeLogData)
        ServiceData.sort_values(by=['DatePDAdjusted'],inplace = True)
        ServiceData.reset_index(inplace = True) 
        
        # print(ServiceData)
        
        CyclesInaDayDict = dict()
        
        for day in list(set(OpsData["DatePDString"].tolist())):
            CyclesInaDayDict[day] = -1
            
        for day in list(set(OpsData["DatePDString"].tolist())):
            subData = OpsData.loc[OpsData['DatePDString'] == day]
            DayCount = subData['DayCount'].tolist()
            CyclesInaDayDict[day] = (sum(DayCount))
        
        RULList = []
        
        for index, row in ServiceData.iterrows():
            # print("INDEX::::",index)
            
            if index < len(ServiceData.index)-1:
                Dates = []
                Cycles = []
                Failure1Frame = ServiceData.iloc[[index]]
                Failure2Frame = ServiceData.iloc[[index+1]]
                
                TotalCyclesAtLog = Failure2Frame['OldValue'].tolist()[0]
                FirstFailureDate = Failure1Frame['DatePD.1'].tolist()[0]
                SecondFailureDate = Failure2Frame['DatePD.1'].tolist()[0]
                
                # print("TotalCyclesAtLog:::::::::::",TotalCyclesAtLog)
                # print("FirstFailureDate:::::::::::",FirstFailureDate)
                # print("SecondFailureDate:::::::::::",SecondFailureDate)
                
                for iday in range (FirstFailureDate,SecondFailureDate+1):
                    if FirstFailureDate != SecondFailureDate:
                        if iday in list(CyclesInaDayDict.keys()):
                            Dates.append(iday)
                            if(iday == FirstFailureDate):
                                Cycles.append(CyclesInaDayDict[iday]*(1 - Failure1Frame['Percentage'].tolist()[0]/100))
                            elif (iday == SecondFailureDate):
                                Cycles.append(CyclesInaDayDict[iday]*(Failure2Frame['Percentage'].tolist()[0]/100))
                            else:
                                Cycles.append(CyclesInaDayDict[iday])
                
                # print("Dates:::",Dates)
                # print("Cycles:::",Cycles)
                
                NumberOfCyclesFromProcess = sum(Cycles)
                if TotalCyclesAtLog != 0:
                    PercentageError = (abs(NumberOfCyclesFromProcess-TotalCyclesAtLog)/TotalCyclesAtLog)*100
                else:
                    PercentageError = 99999
                RULListForPair = []
                
                if len(Cycles) > 1:
                
                    for iterator in range(0,len(Cycles)):
                        if Dates[iterator] == FirstFailureDate:
                            # print(1)
                            RULListForPair.append([ProcessData, Dates[iterator],"FAILURE",TotalCyclesAtLog-sum(Cycles[:iterator+1]),sum(Cycles[iterator+1:]),PercentageError,TotalCyclesAtLog])
                        elif Dates[iterator] == SecondFailureDate:
                            # print(2)
                            RULListForPair.append([ProcessData, Dates[iterator],"FAILURE",TotalCyclesAtLog-sum(Cycles[:iterator+1]),sum(Cycles[iterator+1:]),PercentageError,TotalCyclesAtLog])
                        else:
                            # print(3)
                            RULListForPair.append([ProcessData,Dates[iterator],"BAU",TotalCyclesAtLog-sum(Cycles[:iterator+1]),sum(Cycles[iterator+1:]),PercentageError,TotalCyclesAtLog])
                
                    # print(RULListForPair)
                
                    RULList = RULList + RULListForPair
        for index, row in ServiceData.iterrows():
            Dates = []
            Cycles = []
        
            FailureFrame = ServiceData.iloc[[0]]
            TotalCyclesAtLog = FailureFrame['OldValue'].tolist()[0]
            FirstDate = FailureFrame['DatePD.1'].tolist()[0]
            
            for iBack in range (10,-1,-1):
                if (FirstDate-iBack) in list(CyclesInaDayDict.keys()):
                    Dates.append(FirstDate-iBack)
                    if(FirstDate-iBack == FirstDate):
                        Cycles.append(CyclesInaDayDict[FirstDate-iBack]*(FailureFrame['Percentage'].tolist()[0]/100))
                    else:
                        Cycles.append(CyclesInaDayDict[FirstDate-iBack])
                        
            NumberOfCyclesFromProcess = sum(Cycles)
            if TotalCyclesAtLog != 0:
                PercentageError = (abs(NumberOfCyclesFromProcess-TotalCyclesAtLog)/TotalCyclesAtLog)*100
            else:
                PercentageError = 99999
                
            RULListForPair = []
            
            if len(Cycles) > 1:
                for iterator in range(0,len(Cycles)):
                    if Dates[iterator] == FirstDate:
                        RULListForPair.append([ProcessData, Dates[iterator],"FAILURE",TotalCyclesAtLog-sum(Cycles[:iterator+1]),sum(Cycles[iterator+1:]),PercentageError,TotalCyclesAtLog])
                    else:
                        RULListForPair.append([ProcessData,Dates[iterator],"BAU",TotalCyclesAtLog-sum(Cycles[:iterator+1]),sum(Cycles[iterator+1:]),PercentageError,TotalCyclesAtLog])
            
                RULList = RULList + RULListForPair
                
            break
                
        # print(np.asarray(RULList).shape)
        # print(RULList)
        
        RULListDF = pd.DataFrame(RULList, columns =['Drive','Day','F/BAU','Forward','Backward','PercentageError','LogCount']) 
        # RULListDF = pd.DataFrame(np.asarray(RULList).tolist(), columns =['Drive','Day','F/BAU','Forward','Backward','PercentageError','LogCount']) 
        
        # RULListDF.to_csv("RULListDF.csv", sep=',')
        # print(RULListDF)
        
        return RULListDF
    
    def list_csv_files(directory):
        """
        List all CSV files in the given directory.
        
        :param directory: The path to the directory
        :return: A list of CSV file names
        """
        csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]
        return csv_files
        
    
    Masterdf = pd.DataFrame(columns=['Drive','Day','F/BAU','Forward','Backward','PercentageError','LogCount'])
    Drives = list_csv_files(process_data_path)
    
    for drive in Drives:
        print(failure_data_path + "/" + drive)
        # print("**************************",path.exists(failure_data_path + "/" + drive))
        
        if (path.exists(failure_data_path + "/" + drive) == True):
            print("*************")	
            # print(GetCycles(Tool, Processpath + "/" + drive, failure_data_path + "/" + drive))
            Masterdf = pd.concat([Masterdf,GetCycles(process_data_path + "/" + drive, failure_data_path + "/" + drive)])
            
    return Masterdf
    
## Function for converting RUL into Category
def convert_RUL_to_category(Output_RUL_calculation_csv_path):
    Output_RUL_calculation_csv = pd.read_csv(Output_RUL_calculation_csv_path)
    
    Output_RUL_calculation_csv['average_RUL'] = 0.5*(Output_RUL_calculation_csv['Forward'] + Output_RUL_calculation_csv['Backward'])
    
    Output_RUL_calculation_csv_less_than_15_percent_error = Output_RUL_calculation_csv.loc[Output_RUL_calculation_csv['PercentageError'] <= 8.5]
    
    average_rul_list = Output_RUL_calculation_csv_less_than_15_percent_error['average_RUL'].tolist()
    
    # Convert the list to a 2D array (required by KMeans)
    data_2d = np.array(average_rul_list).reshape(-1, 1)

    # Apply k-means clustering with 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=0).fit(data_2d)

    # Get the cluster labels for each data point
    labels = kmeans.labels_
    
    # Create a list to store the ranges
    ranges = []
    
    # Print the range of each cluster
    for i in range(3):
        cluster_points = [average_rul_list[j] for j in range(len(average_rul_list)) if labels[j] == i]
        cluster_range = [min(cluster_points), max(cluster_points)]
        ranges.append(cluster_range)

    # Sort the ranges list by the first element of each range
    ranges.sort(key=lambda x: x[0])

    # Print the sorted list of ranges
    print("Sorted list of ranges:", ranges)
    
    Low = int(ranges[0][1])
    High = int(ranges[1][1])
    
    print("Low::::::",Low)
    print("High::::::",High)
    
    RULData = Output_RUL_calculation_csv.drop(['average_RUL'], axis=1)
    RULData["Class"] = "YET_TO_BE_FILLED"

    for index, row in RULData.iterrows():	
        if row["F/BAU"] != "FAILURE":
            if row["PercentageError"] < 15:
                if 0.5*(row["Forward"]+row["Backward"]) <= Low:
                    RULData.at[index,"Class"] = 0
                elif 0.5*(row["Forward"]+row["Backward"]) > High:
                    RULData.at[index,"Class"] = 2
                else:
                    RULData.at[index,"Class"] = 1
            else:
                if row["LogCount"] <= Low:
                    RULData.at[index,"Class"] = 0
                elif row["LogCount"] > Low and row["LogCount"] <= High and 0.5*(row["Forward"]+row["Backward"]) > Low:
                    RULData.at[index,"Class"] = 1
                elif row["LogCount"] > High and 0.5*(row["Forward"]+row["Backward"]) > High:
                    RULData.at[index,"Class"] = 2
                else:
                    RULData.at[index,"Class"] = "NA"
        else:
            RULData.at[index,"Class"] = "NA"
            
    return RULData
    
## Function to train the model
def Train_model(rul_file_path):
    MasterSheet = pd.read_csv(rul_file_path)
    
    MasterSheet = MasterSheet.loc[MasterSheet['Class'] != 'NA']
    MasterSheet['Cycles'] = 0.5*(MasterSheet['Forward'] + MasterSheet['Backward'])
    MasterSheet = MasterSheet.drop(['Forward','Backward'], axis=1)
    MasterSheet = MasterSheet.dropna()
    
    train_indices, test_indices = train_test_split(MasterSheet.index, test_size=0.2, random_state=42)
    MasterSheet['Modelling'] = 'Train'
    MasterSheet.loc[test_indices, 'Modelling'] = 'Test'
    
    MasterSheet.to_csv("ft.csv",index=False)
    
    Labels = 3
    SeqLength = 36
    
    ColumnsConsidered = ['DayCount','AvgVelocity','STDVelocity','AvgEnd','STDEnd','AvgPeak','STDPeak','MaxPeak','MinPeak','MAX_PEAK_RATIO','MIN_PEAK_RATIO']
    FeatureLength = len(ColumnsConsidered)
    
    ValidDays = MasterSheet['Day'].tolist()
    Drives = MasterSheet['Drive'].tolist()
    Classes = MasterSheet['Class'].tolist()
    Cycles = MasterSheet['Cycles'].tolist()
    Train_Test_Dec = MasterSheet['Modelling'].tolist()
    
    DatabaseDrives = dict()
    for drive in list(set(Drives)):
        DatabaseDrives[drive] = pd.DataFrame()

    for drive in list(set(Drives)):
        DriveOneYearData = pd.read_csv(drive)
        
        DriveOneYearData["MAX_PEAK_RATIO"] = "YET_TO_BE_FILLED"
        DriveOneYearData["MIN_PEAK_RATIO"] = "YET_TO_BE_FILLED"
        
        DriveOneYearData['MAX_PEAK_RATIO'] = DriveOneYearData["MaxPeak"]/DriveOneYearData["AvgPeak"]
        DriveOneYearData['MIN_PEAK_RATIO'] = DriveOneYearData["MinPeak"]/DriveOneYearData["AvgPeak"]
        
        
        '''for index, row in DriveOneYearData.iterrows():
            DriveOneYearData.at[index,'MAX_PEAK_RATIO'] = row["MaxPeak"]/row["AvgPeak"]
            DriveOneYearData.at[index,'MIN_PEAK_RATIO'] = row["MinPeak"]/row["AvgPeak"]
            
        DriveOneYearData[ColumnsConsidered] = DriveOneYearData[ColumnsConsidered].apply(lambda x: (x - x.min()) / (x.max() - x.min()))'''
        DatabaseDrives[drive] = DriveOneYearData
        
    X = []
    Y = []
    YRaw = []
    DecList = []

    DumpXTest = []
    DumpYTestClassifier = []

    count = 0

    ColumnHeads = []

    for day,drive,cat,cycle,Dec in zip(ValidDays,Drives,Classes,Cycles,Train_Test_Dec):
        driveDF = DatabaseDrives[drive]
        
        ColumnHeads = list(driveDF.columns.values)
        
        driveDFForDay = driveDF.loc[driveDF['DatePDString'] == day]
        
        if Dec == "Test":
            for k in range (0, len(driveDFForDay.values)):
                DumpXTest.append(driveDFForDay.values[k])
            DumpYTestClassifier.append(cat)
        
        npArraydriveDFForDay = driveDFForDay[ColumnsConsidered].copy().values
        npArraydriveDFForDay = np.transpose(npArraydriveDFForDay)
        
        npArraydriveDFForDayList = npArraydriveDFForDay.tolist()
        SeqList = []
        
        for ar in npArraydriveDFForDayList:
            SeqList.append(len(ar))
            for app in range (SeqLength-len(ar)):
                ar.append(0)
        
        if max(SeqList) <= SeqLength:
            X.append(npArraydriveDFForDayList)
            Y.append(cat)
            YRaw.append(cat)
            DecList.append(Dec)
        
        count = count + 1

    # DumpXTestDF = pd.DataFrame(DumpXTest, columns =ColumnHeads) 
    # DumpYTestClassifierDF = pd.DataFrame(DumpYTestClassifier, columns =['Actual']) 

    # DumpXTestDF.to_csv("DumpXTest.csv", sep=',')
    # DumpYTestClassifierDF.to_csv("DumpYTestClassifier.csv", sep=',')

    # input("Press Enter to continue...")
        
    print("Datasets Generated!!")

    X_train = []
    X_test = []
    y_train = []
    y_test = []
    y_train_Classifiers = []
    y_test__Classifiers = []

    for inputDataIndex in range(0, len(X)):
        if DecList[inputDataIndex] == "Train":
            X_train.append(X[inputDataIndex])
            y_train.append(Y[inputDataIndex])
            y_train_Classifiers.append(YRaw[inputDataIndex])
        else:
            X_test.append(X[inputDataIndex])
            y_test.append(Y[inputDataIndex])
            y_test__Classifiers.append(YRaw[inputDataIndex])
            
    
    sample_weights = compute_sample_weight(
                            class_weight='balanced',
                            y=y_train_Classifiers #provide your own target name
                        )

    
    print(sample_weights)
    
    '''param = {'max_depth': 5, 'learning_rate': 0.08156152676629898, 'n_estimators': 730, 'subsample': 0.9280383020148235, 'colsample_bytree': 0.5493854171204564, 'lambda': 3.6999611708703646, 'alpha': 1.2593800013067225, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss'}
    
    param = {'max_depth': 9, 'learning_rate': 0.061690613676626836, 'n_estimators': 916, 'subsample': 0.5825121547508354, 'colsample_bytree': 0.7261034183715559, 'lambda': 1.5095854591811095, 'alpha': 0.17000956365610748, , 'objective': 'multi:softprob', 'eval_metric': 'mlogloss'}'''
    
    param = {'max_depth': 9, 'learning_rate': 0.025429230516274104, 'n_estimators': 856, 'subsample': 0.8844072523132632, 'colsample_bytree': 0.8202154321423809, 'lambda': 0.01897670094343792, 'alpha': 0.2765517695867977, 'objective': 'multi:softprob', 'eval_metric': 'mlogloss'}
    
    clf = xgb.XGBClassifier(**param)
    
    '''np.savetxt("X_train_tune.csv", np.asarray(X_train).reshape(-1,SeqLength*FeatureLength), delimiter=",")
    np.savetxt("y_train_tune.csv", np.asarray(y_train_Classifiers), delimiter=",")
    np.savetxt("X_test_tune.csv", np.asarray(X_test).reshape(-1,SeqLength*FeatureLength), delimiter=",")
    np.savetxt("y_test_tune.csv", np.asarray(y_test__Classifiers), delimiter=",")'''
    

    clf.fit(np.asarray(X_train).reshape(-1,SeqLength*FeatureLength), y_train_Classifiers,sample_weight=sample_weights)
    # clf.fit(X_resampled, y_resampled, sample_weight=sample_weights)

    ypred = clf.predict(np.asarray(X_test).reshape(-1,SeqLength*FeatureLength))
    
    '''np.savetxt('Pred.csv', ypred, delimiter=",") 
    np.savetxt('Actual.csv', y_test__Classifiers, delimiter=",") '''
    
    print(accuracy_score(y_test__Classifiers, ypred))
    print(f1_score(y_test__Classifiers, ypred, average='macro'))
    conf_mat = confusion_matrix(y_test__Classifiers, ypred)
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    # print(conf_mat.diagonal())	
    

convert_csv_to_file_struct("FeaturesLog.csv", "ServiceLog.csv")
    
Output_RUL_calculation_csv = CalculateRUL("SegregatedData/1","DieChangeActionLogSegregation/1")
Output_RUL_calculation_csv.to_csv("Output_RUL_calculation_csv.csv", index = False)

RUL_category_file = convert_RUL_to_category("Output_RUL_calculation_csv.csv")
RUL_category_file.to_csv("RUL_category_file.csv", index = False)

Train_model("RUL_category_file.csv")
