# -*- coding: utf-8 -*-
# Authors: DEVANSH JAIN - devajain
#           SANYAM RAJPAL - srajpal
#           JASHJEET SINGH MADAN - jsmadan
import math
import numpy as np
import pickle
max_depth = 3

def read_images(filename, array):
    with open(filename, "r") as f: 
        for line in f:
            array.append(line.split())
    return(array)  

def testing( test_rgb , d_tree , labels ) :
    test_angles = [ ]
    for i in range( len( test_rgb ) ) :
        j = 0
        k = 0
        while j < len( d_tree ) :
            if d_tree[ j ][ 0 ] == max_depth :
                if test_rgb[ i ][ int( d_tree[ j ][ 1 ] ) ] >= d_tree[ j ][ 2 ] :
                    test_angles.append( labels[ k ] )
                else :
                    test_angles.append( labels[ k + 1 ] )
                j = len( d_tree )
            else :
                if test_rgb[ i ][ int( d_tree[ j ][ 1 ] ) ] >= d_tree[ j ][ 2 ] :
                    j += 1
                    continue
                else :
                    d = d_tree[ j ][ 0 ]
                    k += int( 2 ** ( max_depth - d ) )
                    j += 2
                    while j < len( d_tree ) and d_tree[ j ][ 0 ] != d + 1 :
                        j += 1
                    continue
    return test_angles

def cal_entropy( angles , rgb , mean , index , depth , d_tree , labels ) :
    
    if depth > max_depth :
        x , y = np.unique( angles , return_counts = True )
        result = np.array( [ x[ np.argmax( y ) ] ] * len( angles ) )
        labels.append( result[ 0 ] )
        return result
    
    n = len( rgb )
    m = len( rgb[ 0 ] )
    
    mean_array = rgb.mean( axis = 0 )
    info_gain = [ ]
    for j in range( m ) :
        count_1_0 = count_1_1 = count_1_2 = count_1_3 = count_1 = 0.0
        count_0_0 = count_0_1 = count_0_2 = count_0_3 = count_0 = 0.0
        for i in range( n ) :
            if rgb[ i ][ j ] >= mean_array[ j ] :
                if angles[ i ] == "0" :
                    count_0_0 += 1.0
                elif angles[ i ] == "90" :
                    count_0_1 += 1.0
                elif angles[ i ] == "180" :
                    count_0_2 += 1.0
                else :
                    count_0_3 += 1.0
                count_0 += 1.0
            else :
                if angles[ i ] == "0" :
                    count_1_0 += 1.0
                elif angles[ i ] == "90" :
                    count_1_1 += 1.0
                elif angles[ i ] == "180" :
                    count_1_2 += 1.0
                else :
                    count_1_3 += 1.0
                count_1 += 1.0
                
        p_0 = count_0_0 / count_0
        p_1 = count_0_1 / count_0
        p_2 = count_0_2 / count_0
        p_3 = count_0_3 / count_0
        
        temp = -p_0 * math.log2( p_0 ) - p_1 * math.log2( p_1 ) - p_2 * math.log2( p_2 ) - p_3 * math.log2( p_3 )
        
        p_0 = count_1_0 / count_1
        p_1 = count_1_1 / count_1
        p_2 = count_1_2 / count_1
        p_3 = count_1_3 / count_1
        
        temp += -p_0 * math.log2( p_0 ) - p_1 * math.log2( p_1 ) - p_2 * math.log2( p_2 ) - p_3 * math.log2( p_3 )
        
        info_gain.append( 4.0 - temp )
    
    index = info_gain.index( max( info_gain ) )
    true_rgb = [ ] 
    false_rgb = [ ]
    
    true_angles = [ ]
    false_angles = [ ]

    
    for i in range( n ) :
        if rgb[ i ][ index ] >= mean_array[ index ] :
            true_rgb.append( rgb[ i ] )
            true_angles.append( angles[ i ] )
        else :
            false_rgb.append( rgb[ i ] )
            false_angles.append( angles[ i ] )
            
    true_angles = np.array( true_angles )
    true_rgb = np.array( true_rgb )
    false_angles = np.array( false_angles )
    false_rgb = np.array( false_rgb )
    d_tree.append( ( depth , index , mean_array[ index ] ) )
    true_angles = cal_entropy( true_angles , true_rgb , mean_array[ index ] , index , depth + 1 , d_tree , labels )
    false_angles = cal_entropy( false_angles , false_rgb , mean_array[ index ] , index , depth + 1 , d_tree , labels )
    j = k = 0
    result_angles = [ ]
    
    for i in range( n ) :
        if rgb[ i ][ index ] >= mean_array[ index ] :
            result_angles.append( true_angles[ j ] )
            j += 1
        else :
            result_angles.append( false_angles[ k ] )
            k += 1
    return result_angles 
    

def decTreeTrain(train_filename, model_file):

    images, angles, rgb,  = [] , [] ,[] 
    array = []
    d_tree = [ ]
    labels = [ ]
    array = read_images(train_filename, array)
    for i in range (len(array)):
        images.append(array[i][0])
        angles.append(array[i][1])
        rgb.append(array[i][2:])
    angles = np.array( angles )
    rgb = np.array( [ np.array( l ) for l in rgb ] )
    rgb = rgb.astype( float )
    cal_entropy( angles , rgb , 0 , 0 , 0 , d_tree , labels )
    d_tree = np.array(d_tree)
    labels = np.array(labels)
    
    temp = {'labels' : labels , 'd_tree' : d_tree}
    
    with open(model_file, 'wb') as f:
        pickle.dump(temp, f , protocol = pickle.HIGHEST_PROTOCOL )
  
def decTreeTest(test_filename, model_file):
    test_images, test_angles, test_rgb, test_array = [] , [] ,[], []
    test_array = read_images(test_filename, test_array)    
    for i in range (len(test_array)):
        test_images.append(test_array[i][0])
        test_angles.append(test_array[i][1])
        test_rgb.append(test_array[i][2:])
    test_rgb = np.array( [ np.array( l ) for l in test_rgb ] )
    test_rgb = test_rgb.astype( float )
    test_angles = np.array( test_angles )
    with open(model_file, 'rb') as f:
        v= pickle.load(f)
    x = testing(  test_rgb, np.array( v['d_tree'] ) , np.array( v['labels'] ) )
    with open('output.txt', 'w') as f:
        for i in range(len(x)):
            b = [test_images[i], " ",  x[i], "\n"]
            f.writelines(b)
    x =  np.array( x )
    print( "Accuracy DTree" , (( x == test_angles ).sum( ) / len( test_angles ))*100, "%" )
    
def dtreemain(name, filename, model_file):
    if name=='train':
        decTreeTrain(filename, model_file)
    elif name=='test':
        decTreeTest(filename, model_file)
