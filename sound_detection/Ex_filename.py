import os 


def ex_filename():
    base = '/home/libedev/mute/mute-hero/download/'
    train = 'train/'
    test = 'test/'
    
    count =50
    
    for filename in os.listdir(str(base)):
        
        if filename == 'train':
            
            for file in os.listdir(str(base)+str(train)):
                
                if file == 'total_plane' :
                    
                    for ff in os.listdir(str(base)+str(train)+str(file)):
                        print(ff)
                        count +=1
                        os.renames(str(base)+str(train)+str(file)+"/"+str(ff) , 
                                      str(base)+str(train)+str(file)+"/"+"train"+"_"+"airplane"+"_"+ str(count) +".wav")



def ex_filename_test():
    base = '/home/libedev/mute/mute-hero/download/'
    train = 'train/'
    test = 'test/'
    
    count =20
    
    for filename in os.listdir(str(base)):    
    
        if filename  == 'test':
            
            for file in os.listdir(str(base)+str(test)):
                
                    
                    if file == 'total_plane_test' : # 변환하고 싶은 카테고리명 
                        
                        for ff in os.listdir(str(base)+str(test)+str(file)):
                            count +=1
                            
                            os.renames(str(base)+str(test)+str(file)+"/"+str(ff) , 
                                      str(base)+str(test)+str(file)+"/"+"test"+"_"+"airplane"+"_"+ str(count) +".wav")



if __name__ == "__main__":
    

    ex_filename()
    ex_filename_test()

