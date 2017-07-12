LLC4SimilaritySearch:

This code is the public Matlab script for our paper:
Similarity Search for Image Retrieval via Local-constrained Linear Coding
Wang Mao, et al.
2017

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Our work is based on the following papers :
[1]Efficient Large-Scale Similarity Search Using Matrix Factorization
Ahmet Iscen, Michael Rabbat, and Teddy Furon
CVPR 2016

and 

[2] Locality-constrained Linear Coding for image classification
J. Wang and J. Yang and K. Yu and F. Lv and T. Huang and Y. Gong
CVPR2010

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Data:
     To run the code, we need to the precomputed RMAC feature released in the CVPR2016 paper, 
     which can be downloaded at: ftp://ftp.irisa.fr/local/texmex/corpus/memvec/cvpr16/rmac

Dependency:
    The code need the Yael library for k-means clustring and nearest neighbor search, 
    which can be downloaded at: 
    
Running:
    In matlab, run:
                   wm_run_simple_globalQuery.m    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Limitation:
   1.The proposed approach is only work in Paris6k and Paris106k dataset, where each 
    query has sufficient positive database images.
    For Oxford5k and Oxford105k dataset, the proposed approach will fail as some queries 
    have insufficient positive database images.

   2, As the randomness of k-means clustering, the accuracy is not fixed. However, the average 
    accuracy in Paris dataset is better than the matrix factorization approach in the CVPR2016 paper.
     On the Paris6k dataset, the result is : 
                 mAP_baseline(exhaustive search )       = 0.8476
                 mAP_DL(matrix factorization, CVPR2016) = 0.8707
                 mAP_LLC(ours)                          = 0.8888(std=0.26, ten times running)

If you have any problem, please contact me:
wongwowcs%s%d dot com , %(@,100+60+3)     
