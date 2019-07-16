import os

def rename_file():
    
    base = '/home/libedev/mute/mute-hero/download/'
    train = 'train/'
    test = 'test/'
    
    count = 0
        
    
    for filename in os.listdir(str(base)):
            
        if filename  == 'train':
            
            for file in os.listdir(str(base)+str(train)):
                
                    
                    if file == 'airplane' :
                        for ff in os.listdir(str(base)+str(train)+str(file)):
                            count +=1
                            os.rename(str(base)+str(train)+str(file)+"/"+str(ff) , 
                                      str(base)+str(train)+str(file)+"/"+"train"+"_"+str(file)+"_"+ str(count) +".wav")
                        count =0
                             
                    
                    elif file == 'CarEngine' :
                        for ff in os.listdir(str(base)+str(train)+str(file)):
                            count +=1
                            os.rename(str(base)+str(train)+str(file)+"/"+str(ff) , 
                                      str(base)+str(train)+str(file)+"/"+"train"+"_"+str(file)+"_"+ str(count) +".wav")
                        count =0
                        
                    elif file == 'Conversation' :
                        for ff in os.listdir(str(base)+str(train)+str(file)):
                            count +=1
                            os.rename(str(base)+str(train)+str(file)+"/"+str(ff) , 
                                      str(base)+str(train)+str(file)+"/"+"train"+"_"+str(file)+"_"+ str(count) +".wav")
                        count =0
                        
                    elif file == 'Horn' :
                        for ff in os.listdir(str(base)+str(train)+str(file)):
                            count +=1
                            os.rename(str(base)+str(train)+str(file)+"/"+str(ff) , 
                                      str(base)+str(train)+str(file)+"/"+"train"+"_"+str(file)+"_"+ str(count) +".wav")
                        count =0
                        
                    elif file == 'Laughter' :
                        for ff in os.listdir(str(base)+str(train)+str(file)):
                            count +=1
                            os.rename(str(base)+str(train)+str(file)+"/"+str(ff) , 
                                      str(base)+str(train)+str(file)+"/"+"train"+"_"+str(file)+"_"+ str(count) +".wav")
                        count =0
                        
                    elif file == 'ManSpeech' :
                        for ff in os.listdir(str(base)+str(train)+str(file)):
                            count +=1
                            os.rename(str(base)+str(train)+str(file)+"/"+str(ff) , 
                                      str(base)+str(train)+str(file)+"/"+"train"+"_"+str(file)+"_"+ str(count) +".wav")
                        count =0
                    
                    elif file == 'Noise' :
                        for ff in os.listdir(str(base)+str(train)+str(file)):
                            count +=1
                            os.rename(str(base)+str(train)+str(file)+"/"+str(ff) , 
                                      str(base)+str(train)+str(file)+"/"+"train"+"_"+str(file)+"_"+ str(count) +".wav")
                        count =0
                        
                        
                    elif file == 'Vehicle' :
                        for ff in os.listdir(str(base)+str(train)+str(file)):
                            count +=1
                            os.rename(str(base)+str(train)+str(file)+"/"+str(ff) , 
                                      str(base)+str(train)+str(file)+"/"+"train"+"_"+str(file)+"_"+ str(count) +".wav")
                        count =0
                        
                    else:
                        for ff in os.listdir(str(base)+str(train)+str(file)):
                            count +=1
                            os.rename(str(base)+str(train)+str(file)+"/"+str(ff) , 
                                      str(base)+str(train)+str(file)+"/"+"train"+"_"+"WomanSpeech"+"_"+ str(count) +".wav")
                        count =0
                        
                        
if __name__ == "__main__":
    

    rename_file()                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        