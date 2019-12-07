#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "common.h"
#include <vector>
#include <signal.h>
#include <cmath>
#include <unistd.h>
#include <map>
#include <set>

//
//  benchmarking program
//
using std::vector;
using std::map;
using std::set;

#define cutoff 0.01    //Value copied from common.cpp
#define density 0.0005

double sb, sg;
int cb; //number of bins

void bin_particle(particle_t& p, vector<bin>& b)
{
    int x = p.x / sb;
    int y = p.y / sb;
    //printf("b %d. x %d. y %d", x*cb + y, x, y);
    //fflush(stdout);
    //printf(", size %ld.\n", b[x*cb + y].size());
    b[x*cb + y].push_back(p);
}


inline void get_neighbors(int i, int j, vector<int>& neighbors)
{
    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            if (dx == 0 && dy == 0)
                continue;
            if (i + dx >= 0 && i + dx < cb && j + dy >= 0 && j + dy < cb) {
                int index = (i + dx) * cb + j + dy;
                neighbors.push_back(index);
            }
        }
    }
}


int main( int argc, char **argv )
{    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg; 
 
    //
    //  process command line parameters
    //
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
    
    int n = read_int( argc, argv, "-n", 100000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );

    // Printout the hosts
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Hello from %s\n", processor_name);
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


    particle_t *p = (particle_t*) malloc( n * sizeof(particle_t) );
    
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );

    set_size( n );
    if( rank == 0 )
        init_particles( n, p );

    MPI_Bcast(p, n, PARTICLE, 0, MPI_COMM_WORLD);

    vector<bin> b;
   
    sg = sqrt(n * density);
    sb = cutoff;  
    cb = int(sg / sb) + 1; // Should be around sqrt(N/2)

    // printf("size of grid: %.4lf\n", sg);
    // printf("Number of Bins: %d*%d\n", cb, cb);
    // printf("Bin Size: %.2lf\n", sb);
    // Increase\Decrease bin_count to be something like 2^k?
    
    b.resize(cb * cb);

    for (int i = 0; i < n; i++)
    {
        int x = int(p[i].x / sb);
        int y = int(p[i].y / sb);
        b[x*cb + y].push_back(p[i]);
    }

    delete[] p;
    p = NULL;

    int x_bins_per_proc = cb / n_proc;

    // although each worker has all particles, we only access particles within
    // my_bins_start, my_bins_end.

    int my_bins_start = x_bins_per_proc * rank;
    int my_bins_end = x_bins_per_proc * (rank + 1);

    if (rank == n_proc - 1)
        my_bins_end = cb;
    
    // printf("worker %d: from %d to %d.\n", rank, my_bins_start, my_bins_end);
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;

        // if( find_option(argc, argv, "-no" ) == -1 )
        //     if( fsave && (step%SAVEFREQ) == 0 )
        //         save( fsave, n, particles );

        // compute local forces
        for (int i = my_bins_start; i < my_bins_end; ++i) {
            for (int j = 0; j < cb; ++j) 
            {
                bin& vect = b[i * cb + j];

                for (int k = 0; k < vect.size(); k++)
                   vect[k].ax = vect[k].ay = 0;

                for (int dx = -1; dx <= 1; dx++)   //Search over nearby 8 bins and itself
                {       
                  for (int dy = -1; dy <= 1; dy++)
                  {
                    if (i + dx >= 0 && i + dx < cb && j + dy >= 0 && j + dy < cb)
                    {
                        bin& vect2 = b[(i+dx) * cb + j + dy];
                        for (int k = 0; k < vect.size(); k++)
                          for (int l = 0; l < vect2.size(); l++)
                             apply_force( vect[k], vect2[l], &dmin, &davg, &navg);
                    }     
                  }
                }

            }
        }

        if (find_option( argc, argv, "-no" ) == -1) {
          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
          if (rank == 0){
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) 
                absmin = rdmin;
          }
        }

        // move, but not rebin
        bin local_move;
        bin remote_move;

        for (int i = my_bins_start; i < my_bins_end; ++i) {
            for (int j = 0; j < cb; ++j) {
                bin& bin = b[i * cb + j];
                int f = bin.size(), k = 0;
                for (; k < f; ) {
                    move(bin[k]);
                    int x = int(bin[k].x / sb);
                    int y = int(bin[k].y / sb);
                    if (my_bins_start <= x && x < my_bins_end) {
                        if (x == i && y == j)
                            ++k;
                        else {
                            local_move.push_back(bin[k]);
                            bin[k] = bin[--f];
                        }
                    } else {
                        //int who = x / x_bins_per_proc;
                        remote_move.push_back(bin[k]);
                        bin[k] = bin[--f];
                    }
                }
                bin.resize(k);
            }
        }

        for (int i = 0; i < local_move.size(); ++i) {
            bin_particle(local_move[i], b);
        }

        if (rank != 0) {
            for (int i = my_bins_start - 1, j = 0; j < cb; ++j) {
                bin& bin = b[i * cb + j];
                bin.clear();
            }
            for (int i = my_bins_start, j = 0; j < cb; ++j) {
                bin& bin = b[i * cb + j];
                remote_move.insert(remote_move.end(), bin.begin(), bin.end());
                bin.clear();
            }
        }

        if (rank != n_proc - 1) {
            for (int i = my_bins_end, j = 0; j < cb; ++j) {
                bin& bin = b[i * cb + j];
                bin.clear();
            }
            for (int i = my_bins_end - 1, j = 0; j < cb; ++j) {
                bin& bin = b[i * cb + j];
                remote_move.insert(remote_move.end(), bin.begin(), bin.end());
                bin.clear();
            }
        }

        // int len_ = remote_move.size();
        // int total_ = 0;
        // MPI_Reduce(&len_, &total_, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        bin incoming_move;
        int send_count = remote_move.size();
        int recv_counts[n_proc];

        // printf("worker: %d. MPI_Gather.\n", rank);
        MPI_Gather(&send_count, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // now root knows recv_counts

        int displs[n_proc];
        int total_num = 0;

        if (rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < n_proc; ++i) {
                displs[i] = displs[i-1] + recv_counts[i-1];
            }
            total_num = recv_counts[n_proc-1] + displs[n_proc-1];
            // printf("worker: %d, 1. %d / %d.\n", rank, total_, total_num);
            // assert(total_ == total_num);
            incoming_move.resize(total_num);
        }

        // now root knows total_num.

        //printf("worker: %d. MPI_Gatherv.\n", rank);

        MPI_Gatherv(remote_move.data(), send_count, PARTICLE, 
            incoming_move.data(), recv_counts, displs, PARTICLE, 
            0, MPI_COMM_WORLD);

        //printf("worker: %d. Classify.\n", rank);

        vector<bin> scatter_particles;
        scatter_particles.resize(n_proc);

        if (rank == 0) {
            for (int i = 0; i < incoming_move.size(); ++i) {
                int x = int(incoming_move[i].x / sb);

                assert(incoming_move[i].x >= 0 && incoming_move[i].y >= 0 &&
                    incoming_move[i].x <= sg && incoming_move[i].y <= sg);

                int who = min(x / x_bins_per_proc, n_proc-1);
                scatter_particles[who].push_back(incoming_move[i]);

                int row = x % x_bins_per_proc;
                if (row == 0 && who != 0)
                    scatter_particles[who - 1].push_back(incoming_move[i]);
                if (row == x_bins_per_proc-1 && who != n_proc-1)
                    scatter_particles[who + 1].push_back(incoming_move[i]);
            }
            for (int i = 0; i < n_proc; ++i) {
                recv_counts[i] = scatter_particles[i].size();
            }
            displs[0] = 0;
            for (int i = 1; i < n_proc; ++i) {
                displs[i] = displs[i-1] + recv_counts[i-1];
            }
        }

        // printf("worker: %d. MPI_Scatter.\n", rank);
        send_count = 0;
        MPI_Scatter(recv_counts, 1, MPI_INT, &send_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        bin outgoing_move;
        outgoing_move.resize(send_count);

        bin scatter_particles_flatten;
        for (int i = 0; i < scatter_particles.size(); ++i) {
            scatter_particles_flatten.insert(scatter_particles_flatten.end(),
                scatter_particles[i].begin(), scatter_particles[i].end());
        }

        // printf("worker: %d. MPI_Scatterv.\n", rank);
        MPI_Scatterv(scatter_particles_flatten.data(), recv_counts, displs, PARTICLE, 
            outgoing_move.data(), send_count, PARTICLE, 0, MPI_COMM_WORLD);

        // int total__ = 0;
        // MPI_Reduce(&send_count, &total__, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        // if (rank == 0) {
        //     assert(total_ == total__);
        // }

        // printf("worker: %d. Bin.\n", rank);
        for (int i = 0; i < send_count; ++i) {
            particle_t &p1 = outgoing_move[i];
            assert(p1.x >= 0 && p1.y >= 0 && p1.x <= sg && p1.y <= sg);
            bin_particle(p1, b);
        }
        }
    simulation_time = read_timer( ) - simulation_time;
  
    if (rank == 0) {  
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

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
      if( fsum)
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
  
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}

