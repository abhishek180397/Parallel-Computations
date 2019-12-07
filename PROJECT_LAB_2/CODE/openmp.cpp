#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include "omp.h"
#include <vector>

using std::vector;

#define cutoff 0.01    //Values are from common.cpp
#define density 0.0005

double sb,sg;
int nb;

//
//  benchmarking program
//
int main( int argc, char **argv )
{   
    int navg,nabsavg=0,numthreads; 
    double dmin, absmin=1.0,davg,absavg=0.0;
	
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" ); 
        printf( "-no turns off all correctness checks and particle output\n");   
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname ? fopen ( sumname, "a" ) : NULL;      

    particle_t *p = (particle_t*) malloc( n * sizeof(particle_t) );
    set_size( n );
    init_particles( n, p );

    vector<bin> pb;
    vector<bin> temp;
    
    sb = cutoff * 2; 
    sg = sqrt(n*density);
    nb = int(sg / sb)+1; // Should be around sqrt(N/2)

    // printf("size of grid: %.4lf\n",sg);
    // printf("Number of Bins: %d*%d\n",nb,nb);
    // printf("size of bin: %.2lf\n",sb);
    // Increase or Decrease number of bins to be something like 2^k?
    
    pb.resize(nb * nb);

    for (int i = 0; i < n; i++)
    {
        int x = int(p[i].x / sb);
        int y = int(p[i].y / sb);
        pb[x*nb + y].push_back(p[i]);
    }


    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );

    #pragma omp parallel private(dmin) 
    {
        #pragma omp master
        {
            numthreads = omp_get_num_threads();
            temp.resize(numthreads);
        }
        
        for (int step = 0; step < NSTEPS; step++ )
        {
            //#pragma omp parallel private(dmin) 
            {
                navg = 0;
                davg = 0.0;
                dmin = 1.0;
                #pragma omp for reduction (+:navg) reduction(+:davg)
                for (int i = 0; i < nb; i++)
                {
                    for (int j = 0; j < nb; j++)
                    {
                        bin& vect = pb[i*nb+j];
                        for (int k = 0; k < vect.size(); k++)
                            vect[k].ax = vect[k].ay = 0;
                        for (int dx = -1; dx <= 1; dx++)   //Searching nearby 8 bins and itself
                        {
                            for (int dy = -1; dy <= 1; dy++)
                            {
                                if (i + dx >= 0 && i + dx < nb && j + dy >= 0 && j + dy < nb)
                                {
                                    bin& vect2 = pb[(i+dx) * nb + j + dy];
                                    for (int k = 0; k < vect.size(); k++)
                                        for (int l = 0; l < vect2.size(); l++)
                                            apply_force( vect[k], vect2[l], &dmin, &davg, &navg);
                                }
                            }
                        }
                    }
                }
            
                    
                //#pragma omp master
                {
                    int q = omp_get_thread_num();  // Each thread has a seperate tmp vector
                    bin& tp = temp[q];
                    tp.clear();
                    #pragma omp for
                    for (int i = 0; i < nb; i++)
                    {
                        for(int j = 0; j < nb; j++)
                        {
                            bin& vect = pb[i * nb + j];
                            int f = vect.size(), k = 0;
                            for(; k < f; )
                            {
                                move( vect[k] );
                                int x = int(vect[k].x / sb);  //Check the position
                                int y = int(vect[k].y / sb);
                                if (x == i && y == j)  // Still inside original bin
                                    k++;
                                else
                                {
                                    tp.push_back(vect[k]);  // Store paricles that have changed bin. 
                                    vect[k] = vect[--f]; //Remove it from the current bin.
                                }
                            }
                            vect.resize(k);
                        }
                    }
                    //Scan over all tmp vectors using one threads...
                    //Using multiple threads with critical area seems to decrease performance 
                    #pragma omp master 
                    {
                        for(int j=0;j<numthreads;j++)
                        {
                            bin& tp = temp[j];
                            for (int i = 0; i < tp.size(); i++)  // Put them into the new bin 
                            {
                                int x = int(tp[i].x / sb);
                                int y = int(tp[i].y / sb);
                                //If using multiple threads, below is critical area
                                //#pragma omp critical  
                                pb[x*nb+y].push_back(tp[i]);
                            }
                        }
                    }
                    
                }
            }
            
            if( find_option( argc, argv, "-no" ) == -1 )
            {
                #pragma omp master
                if (navg) 
                {
                    absavg +=  davg/navg;
                    nabsavg++;
                }
                  
                #pragma omp critical
                if (dmin < absmin) 
                    absmin = dmin;
                    
                #pragma omp master
                if( fsave && (step%SAVEFREQ) == 0 )
                    save( fsave, n, p );
            }
            
            #pragma omp barrier // Wait for all threads (mostly the master) to finish

        }
    }
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "NThreads: %d, n = %d, simulation time = %g seconds", numthreads,n, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
        if (nabsavg) absavg /= nabsavg;
        // 
        //  -The minimum distance absmin between 2 particles during the run of the simulation
        //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
        //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
        //
        //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
        //
        printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
        if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
        if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");     

    //
    // Printing summary data
    //
    if( fsum ) 
        fprintf(fsum,"%d %g\n",n,simulation_time);
 
    //
    // Clearing space
    //
    if( fsum )
        fclose( fsum );

    free( p );
    
    if( fsave )
        fclose( fsave );
    
    return 0;
}
