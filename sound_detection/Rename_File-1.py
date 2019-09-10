import os

def rename_file():
    
    base = '/home/libedev/mute/mute-hero/download/dataset/'
    train = 'new_train/'
    test = 'new_test/'
    
    count = 0
        
    
    for filename in os.listdir(str(base)):
            
        if filename  == 'new_train':
            
            for file in os.listdir(str(base)+str(train)):
                
                    
                    if file == 'ManSpeech' :   # 변환하고 싶은 카테고리명 
                        
                        for ff in os.listdir(str(base)+str(train)+str(file)):
                            count +=1
                            
                            os.rename(str(base)+str(train)+str(file)+"/"+str(ff) , 
                                      str(base)+str(train)+str(file)+"/"+"test"+"_"+str(file)+"_"+ str(count) +".wav")
                        count =0
                        
                        
                        
        elif filename  == 'new_test':
            
            for file in os.listdir(str(base)+str(train)):
                
                    
                    if file == 'ManSpeech' : # 변환하고 싶은 카테고리명 
                        
                        for ff in os.listdir(str(base)+str(test)+str(file)):
                            count +=1
                            
                            os.rename(str(base)+str(test)+str(file)+"/"+str(ff) , 
                                      str(base)+str(test)+str(file)+"/"+"test"+"_"+str(file)+"_"+ str(count) +".wav")
                        count =0
            else:
                
                pass
                        
                        
                        
if __name__ == "__main__":
    

    rename_file()