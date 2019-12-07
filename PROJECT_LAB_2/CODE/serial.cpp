#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include <vector>

using std::vector;

#define cutoff 0.01    //Values from common.cpp
#define density 0.0005

int nb; //number of bins
double sb,sg; //size of bin,size of grid

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg,nabsavg=0;
    double davg,dmin, absmin=1.0, absavg=0.0;

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
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
    bin temp;

    sb = cutoff * 2; 
    sg = sqrt(n*density); 
    nb = int(sg / sb)+1; // Should be around sqrt(N/2)

    // printf("size of grid: %.4lf\n",sg);
    // printf("Number of Bins: %d*%d\n",nb,nb);
    // printf("size of bin: %.2lf\n",sb);
    // Increase\Decrease binNum to be something like 2^k?
    
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
	
    for( int step = 0; step < NSTEPS; step++ )
    {
	navg = 0;
    davg = 0.0;
	dmin = 1.0;
        //
        //  compute forces
        //
        for (int i = 0; i < nb; i++)
        {
            for (int j = 0; j < nb; j++)
            {
                bin & vect = pb[i*nb+j];
                for (int k = 0; k < vect.size(); k++)
                    vect[k].ax = vect[k].ay = 0;
                for (int dx = -1; dx <= 1; dx++)   //Search over nearby 8 bins and itself
                {
                    for (int dy = -1; dy <= 1; dy++)
                    {
                        if (i + dx >= 0 && i + dx < nb && j + dy >= 0 && j + dy < nb)
                        {
                            bin & vect2 = pb[(i+dx) * nb + j + dy];
                            for (int k = 0; k < vect.size(); k++)
                                for (int l = 0; l < vect2.size(); l++)
                                    apply_force( vect[k], vect2[l], &dmin, &davg, &navg);
                        }
                    }
                }
            }
        }
        
        for (int i = 0; i < nb; i++)
        {
            for(int j = 0; j < nb; j++)
            {
                bin& vect = pb[i * nb + j];
                int z = vect.size(), k = 0;
                for(; k < z; )
                {
                    move( vect[k] );
                    int x = int(vect[k].x / sb); 
                    int y = int(vect[k].y / sb);
                    if (x == i && y == j)  
                        k++;
                    else
                    {
                        temp.push_back(vect[k]);  
                        vect[k] = vect[--z]; 
                    }
                }
                vect.resize(k);
            }
        }
        //
        //  move particles
        //
        for (int i = 0; i < temp.size(); i++)  
        {
            int x = int(temp[i].x / sb);
            int y = int(temp[i].y / sb);
            pb[x*nb+y].push_back(temp[i]);
        }
        temp.clear();
        

        if( find_option( argc, argv, "-no" ) == -1 )
        {
          //
          // Computing statistical data
          //
          if (navg) {
            absavg +=  davg/navg;
            nabsavg++;
          }
          if (dmin < absmin) absmin = dmin;
		
          //
          //  save if necessary
          //
          if( fsave && (step%SAVEFREQ) == 0 )
              save( fsave, n, p );
        }
    }
    simulation_time = read_timer( ) - simulation_time;
    
    printf( "n = %d, simulation time = %g seconds", n, simulation_time);

    if( find_option( argc, argv, "-no" ) == -1 )
    {
      if (nabsavg) absavg /= nabsavg;
    // 
    //  During the simulation, the min distance between the two particles 
    //  A simulation which is correct will have particles at a distance > 0.4 * cutoff with typical values between .7 and .8
    //  Particle's simulation which do not interact correctly will stay a distance < 0.4 * cutoff with typical values between .01 and .05
    //
    //  The absavg(average distance) when most particles are interacting correctly is approx 0.95 and when no particles are interacting is approx 0.66
    //
    printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
    if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
    if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
    }
    printf("\n");     

    //
    // Printing summary data
    //
    if( fsum) 
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
