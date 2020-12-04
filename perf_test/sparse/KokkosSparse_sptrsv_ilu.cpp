/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include "Kokkos_Random.hpp"
#include "KokkosKernels_SparseUtils.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosSparse_spiluk.hpp"

#include "KokkosSparse_sptrsv.hpp"
#include "KokkosSparse_sptrsv_supernode.hpp"

#if defined( KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA )         && \
  (!defined(KOKKOS_ENABLE_CUDA) || (8000 <= CUDA_VERSION)) && \
    defined(KOKKOSKERNELS_INST_DOUBLE)

#if defined(KOKKOSKERNELS_ENABLE_SUPERNODAL_SPTRSV)

#include "KokkosSparse_sptrsv_aux.hpp"

#ifdef KOKKOSKERNELS_ENABLE_TPL_METIS
#include "metis.h"
#endif

using namespace KokkosKernels;
using namespace KokkosKernels::Impl;
using namespace KokkosKernels::Experimental;
using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
using namespace KokkosSparse::PerfTest::Experimental;

enum {CUSPARSE, SUPERNODAL_NAIVE, SUPERNODAL_ETREE, SUPERNODAL_DAG, SUPERNODAL_SPMV, SUPERNODAL_SPMV_DAG};

/* ========================================================================================= */
template<typename scalar_type>
int test_sptrsv_perf (std::vector<int> tests, bool verbose, std::string& matrix_filename,
                      bool metis, int expand_fact, bool merge, bool invert_diag, bool invert_offdiag, bool u_in_csr, int loop) {

  using ordinal_type = int;
  using size_type    = int;
  using STS = Kokkos::Details::ArithTraits<scalar_type>;
  using mag_type = typename STS::mag_type;

  // Default spaces
  //using execution_space = Kokkos::OpenMP;
  using execution_space = Kokkos::DefaultExecutionSpace;
  using memory_space = typename execution_space::memory_space;

  // Host spaces
  using host_execution_space = Kokkos::DefaultHostExecutionSpace;
  using host_memory_space = typename host_execution_space::memory_space;

  //
  using KernelHandle =  KokkosKernels::Experimental::KokkosKernelsHandle
                        <size_type, ordinal_type, scalar_type, execution_space, memory_space, memory_space >;
  using ILUKernelHandle =  KokkosKernels::Experimental::KokkosKernelsHandle
                           <size_type, ordinal_type, scalar_type, host_execution_space, host_memory_space, host_memory_space >;

  //
  using host_crsmat_t = typename KernelHandle::SPTRSVHandleType::host_crsmat_t;
  using      crsmat_t = typename KernelHandle::SPTRSVHandleType::crsmat_t;

  //
  using host_scalar_view_t = Kokkos::View< scalar_type*, host_memory_space >;
  using      scalar_view_t = Kokkos::View< scalar_type*,      memory_space >;

  const scalar_type ZERO (0.0);
  const scalar_type ONE (1.0);

  // tolerance
  mag_type tol = STS::epsilon();

  int num_failed = 0;
  std::cout << std::endl;
  std::cout << "Execution space: " << execution_space::name () << std::endl;
  std::cout << "Memory space   : " << memory_space::name () << std::endl;
  std::cout << std::endl;
  {
    // ==============================================
    // read the CRS matrix ** on host **
    // it stores the supernodal triangular matrix, stored by blocks with explicit zeros
    std::cout << " Supernode Tester Begin:" << std::endl;
    std::cout << " > Read a matrix filename " << matrix_filename << std::endl;
    host_crsmat_t M = KokkosKernels::Impl::read_kokkos_crst_matrix<host_crsmat_t> (matrix_filename.c_str());
    const size_type nrows = M.graph.numRows ();
    std::cout << " nrows = " << nrows << " nnz = " << M.graph.row_map(nrows) << std::endl;

    if (metis) {
      #ifdef KOKKOSKERNELS_ENABLE_TPL_METIS
      auto original_graph   = M.graph; // in_graph
      auto original_row_map = original_graph.row_map;
      auto original_entries = original_graph.entries;
      auto original_values  = M.values;

      idx_t n = original_graph.numRows();
      idx_t nnz = original_row_map(n);
      const idx_t nnzA = original_row_map(n);
      
      // removing diagonals (and casting to METIS idx_t)
      idx_t *metis_rowptr = new idx_t[n+1];
      idx_t *metis_colind = new idx_t[nnz];

      nnz = 0;
      metis_rowptr[0] = 0;
      for (int i = 0; i < n; i++) {
        for (int k = original_row_map(i); k < original_row_map(i+1); k++) {
          if (original_entries(k) != i) {
            metis_colind[nnz] = original_entries(k);
            nnz ++;
          }
        }
        metis_rowptr[i+1] = nnz;
      }

      idx_t *perm = new idx_t[n];
      idx_t *iperm = new idx_t[n];

      std::cout << " calling METIS_NodeND: (n=" << n << ", nnz=" << nnzA << "->" << nnz << ") " << std::endl;
      if (METIS_OK != METIS_NodeND(&n, metis_rowptr, metis_colind, NULL, NULL, perm, iperm)) {
        std::cout << std::endl << "METIS_NodeND failed" << std::endl << std::endl;
      }
      //for (idx_t i = 0; i < n; i++) printf("%ld %ld %ld\n",i, perm[i], iperm[i]);
      //for (idx_t i = 0; i < n; i++) perm[i] = iperm[i] = i;

      using host_graph_t = typename host_crsmat_t::StaticCrsGraphType;
      using host_row_map_view_t = typename host_graph_t::row_map_type::non_const_type;
      using host_cols_view_t    = typename host_graph_t::entries_type::non_const_type;
      using host_values_view_t  = typename host_crsmat_t::values_type::non_const_type;

      host_row_map_view_t hr ("rowmap_view", n+1);
      host_cols_view_t    hc ("colmap_view", nnzA);
      host_values_view_t  hv ("values_view", nnzA);

      nnz = 0; hr (0) = 0;
      for (idx_t i = 0; i < n; i++) {
        for (idx_t j=original_row_map(perm[i]); j < original_row_map(perm[i]+1); j++) {
          hc(nnz) = iperm[ original_entries(j) ];
          hv(nnz) = original_values(j);
          nnz ++;
        }
        hr(1+i) = nnz;
      }
      host_graph_t host_static_graph(hc, hr);
      M = host_crsmat_t("CrsMatrix", nrows, hv, host_static_graph);

      delete [] perm;
      delete [] iperm;
      delete [] metis_rowptr;
      delete [] metis_colind;
      #else
      std::cout << std::endl << " ** TPL_METIS is not enabled ** " << std::endl << std::endl;
      #endif
    }

    // Call ILU(k)
    int fill_level = 1;
    int nnzA = M.graph.row_map(nrows);
    int nnzLU =expand_fact*nnzA*(fill_level+1);

    // > create handle and allocate L/U
    ILUKernelHandle khILU;
    khILU.create_spiluk_handle(SPILUKAlgorithm::SEQLVLSCHD_RP, nrows, nnzLU, nnzLU);

    using host_graph_t = typename host_crsmat_t::StaticCrsGraphType;
    using row_map_view_t = typename host_graph_t::row_map_type::non_const_type;
    using    cols_view_t = typename host_graph_t::entries_type::non_const_type;
    using  values_view_t = typename host_crsmat_t::values_type::non_const_type;
    using in_row_map_view_t = typename host_graph_t::row_map_type;
    using in_cols_view_t    = typename host_graph_t::entries_type;
    using in_values_view_t  = typename host_crsmat_t::values_type;

    row_map_view_t L_row_map("L_row_map", nrows + 1);
    cols_view_t    L_entries("L_entries", khILU.get_spiluk_handle()->get_nnzL());
    values_view_t  L_values ("L_values",  khILU.get_spiluk_handle()->get_nnzL());
    row_map_view_t U_row_map("U_row_map", nrows + 1);
    cols_view_t    U_entries("U_entries", khILU.get_spiluk_handle()->get_nnzU());
    values_view_t  U_values ("U_values",  khILU.get_spiluk_handle()->get_nnzU());

    // > call symbolic ILU
    std::cout << " > Call SpILUk::symbolic with ";
    khILU.get_spiluk_handle()->print_algorithm();
    spiluk_symbolic( &khILU, fill_level,
                     M.graph.row_map, M.graph.entries,
                     L_row_map, L_entries, U_row_map, U_entries );
    Kokkos::resize(L_entries, khILU.get_spiluk_handle()->get_nnzL()); std::cout << " >> resized <<" << std::endl << std::endl;
    Kokkos::resize(L_values,  khILU.get_spiluk_handle()->get_nnzL()); std::cout << " >> resized <<" << std::endl << std::endl;
    Kokkos::resize(U_entries, khILU.get_spiluk_handle()->get_nnzU()); std::cout << " >> resized <<" << std::endl << std::endl;
    Kokkos::resize(U_values,  khILU.get_spiluk_handle()->get_nnzU()); std::cout << " >> resized <<" << std::endl << std::endl;

    // > call numeric ILU
    std::cout << " > Call SpILUk::numeric" << std::endl;
    spiluk_numeric( &khILU, fill_level, M.graph.row_map, M.graph.entries, M.values,
                    L_row_map, L_entries, L_values, U_row_map, U_entries, U_values );

    // > store L/U in csr
    host_crsmat_t L_csr("L", nrows, nrows, khILU.get_spiluk_handle()->get_nnzL(), L_values,  L_row_map,  L_entries);
    host_crsmat_t U_csr("U", nrows, nrows, khILU.get_spiluk_handle()->get_nnzU(), U_values,  U_row_map,  U_entries);

    // > store L in csc
    row_map_view_t Lt_row_map("Lt_row_map", nrows + 1);
    cols_view_t    Lt_entries("Lt_entries", L_row_map(nrows));
    values_view_t  Lt_values ("Lt_values",  L_row_map(nrows));
    transpose_matrix <in_row_map_view_t, in_cols_view_t, in_values_view_t,
                         row_map_view_t,    cols_view_t,    values_view_t,
                         row_map_view_t, host_execution_space>
      (nrows, nrows, L_row_map,  L_entries,  L_values,
                     Lt_row_map, Lt_entries, Lt_values);

    host_crsmat_t L_csc("L", nrows, nrows, khILU.get_spiluk_handle()->get_nnzL(), Lt_values, Lt_row_map, Lt_entries);

    #if 0
    printf( " M = [\n" );
    for (int i = 0; i < nrows; i++) {
      for (int k = M.graph.row_map[i]; k < M.graph.row_map[i+1]; k++) printf( "%d %d %e\n",i,M.graph.entries(k),M.values(k) );
    }
    printf( "];\n" );
    printf( " L = [\n" );
    for (int i = 0; i < nrows; i++) {
      for (int k = L_csc.graph.row_map[i]; k < L_csc.graph.row_map[i+1]; k++) printf( "%d %d %e\n",i,L_csc.graph.entries(k),L_csc.values(k) );
    }
    printf( "];\n" );
    printf( " U = [\n" );
    for (int i = 0; i < nrows; i++) {
      for (int k = U_csr.graph.row_map[i]; k < U_csr.graph.row_map[i+1]; k++) printf( "%d %d %e\n",i,U_csr.graph.entries(k),U_csr.values(k) );
    }
    printf( "];\n" );
    #endif

    // supercols of size one
    int nsuper = nrows;
    Kokkos::View<int*, Kokkos::HostSpace> supercols ("supercols", 1+nsuper);
    int *etree = NULL;
    for (int i = 0; i <= nsuper; i++) {
      supercols (i) = i;
    }

    Kokkos::Timer timer;
    // ==============================================
    // Run all requested algorithms
    for ( auto test : tests ) {
      std::cout << "\ntest = " << test << std::endl;

      KernelHandle khL;
      KernelHandle khU; // TODO: can I just make a copy later (khU = khL)?
      switch(test) {
        case SUPERNODAL_NAIVE:
        case SUPERNODAL_ETREE:
        case SUPERNODAL_DAG:
        case SUPERNODAL_SPMV:
        case SUPERNODAL_SPMV_DAG:
        {
          // ==============================================
          // create an handle
          if (test == SUPERNODAL_NAIVE) {
            std::cout << " > create handle for SUPERNODAL_NAIVE" << std::endl << std::endl;
            khL.create_sptrsv_handle (SPTRSVAlgorithm::SUPERNODAL_NAIVE, nrows, true);
            khU.create_sptrsv_handle (SPTRSVAlgorithm::SUPERNODAL_NAIVE, nrows, false);
          } else if (test == SUPERNODAL_DAG) {
            std::cout << " > create handle for SUPERNODAL_DAG" << std::endl << std::endl;
            khL.create_sptrsv_handle (SPTRSVAlgorithm::SUPERNODAL_DAG, nrows, true);
            khU.create_sptrsv_handle (SPTRSVAlgorithm::SUPERNODAL_DAG, nrows, false);
          } else if (test == SUPERNODAL_SPMV_DAG) {
            std::cout << " > create handle for SUPERNODAL_SPMV_DAG" << std::endl << std::endl;
            khL.create_sptrsv_handle (SPTRSVAlgorithm::SUPERNODAL_SPMV_DAG, nrows, true);
            khU.create_sptrsv_handle (SPTRSVAlgorithm::SUPERNODAL_SPMV_DAG, nrows, false);
          }
          khL.set_sptrsv_use_full_dag (true);
          khU.set_sptrsv_use_full_dag (true);

          // verbose (optional, default is false)
          khL.set_sptrsv_verbose (verbose);
          khU.set_sptrsv_verbose (verbose);

          // specify wheather to merge supernodes (optional, default merge is false)
          std::cout << " Merge Supernode    : " << merge << std::endl;
          khL.set_sptrsv_merge_supernodes (merge);
          khU.set_sptrsv_merge_supernodes (merge);

          // specify wheather to invert diagonal blocks
          std::cout << " Invert diagonal    : " << invert_diag << std::endl;
          khL.set_sptrsv_invert_diagonal (invert_diag);
          khU.set_sptrsv_invert_diagonal (invert_diag);

          // specify wheather to apply diagonal-inversion to off-diagonal blocks (optional, default is false)
          std::cout << " Invert Off-diagonal: " << invert_offdiag << std::endl;
          khL.set_sptrsv_invert_offdiagonal (invert_offdiag);
          khU.set_sptrsv_invert_offdiagonal (invert_offdiag);
 
          // ==============================================
          // do symbolic analysis (preprocssing, e.g., merging supernodes, inverting diagonal/offdiagonal blocks,
          // and scheduling based on graph/dag)
          // note: here, we use U = L'
          khU.get_sptrsv_handle ()->set_column_major (!khL.get_sptrsv_handle ()->is_column_major ());
          sptrsv_supernodal_symbolic (nsuper, supercols.data (), etree, L_csc.graph, &khL, U_csr.graph, &khU);

          // ==============================================
          // do numeric compute (copy numerical values from L to our sptrsv data structure)
          sptrsv_compute (&khL, L_csc);
          sptrsv_compute (&khU, U_csr);

          {
            // ==============================================
            // Preaparing for the first solve with L
            //> create the known solution and set to all 1's ** on host **
            host_scalar_view_t sol_host ("sol_host", nrows);
            //Kokkos::deep_copy (sol_host, ONE);
            Kokkos::Random_XorShift64_Pool<host_execution_space> random(13718);
            Kokkos::fill_random(sol_host, random, scalar_type(1));

            // > create the rhs ** on host **
            // A*sol generates rhs: rhs is dense, use spmv
            host_scalar_view_t rhs_host ("rhs_host", nrows);
            KokkosSparse::spmv ( "N", ONE, L_csr, sol_host, ZERO, rhs_host);

            // ==============================================
            // copy rhs to the default host/device
            scalar_view_t rhs ("rhs", nrows);
            scalar_view_t sol ("sol", nrows);
            Kokkos::deep_copy (rhs, rhs_host);

            // ==============================================
            // do L solve
            timer.reset();
            sptrsv_solve (&khL, sol, rhs);
            Kokkos::fence();
            std::cout << " > Lower-TRI: " << std::endl;
            std::cout << "   Solve Time   : " << timer.seconds() << std::endl;

            // copy solution to host
            Kokkos::deep_copy (sol_host, sol);

            // ==============================================
            // Error Check ** on host **
            Kokkos::fence ();
            std::cout << std::endl;
            if (!check_errors (tol, L_csr, rhs_host, sol_host)) {
              num_failed ++;
            }

            // Benchmark
            // L-solve
            double min_time = 1.0e32;
            double max_time = 0.0;
            double ave_time = 0.0;
            Kokkos::fence ();
            for(int i = 0; i < loop; i++) {
              timer.reset();
              sptrsv_solve (&khL, sol, rhs);
              Kokkos::fence();
              double time = timer.seconds ();
              ave_time += time;
              if(time > max_time) max_time = time;
              if(time < min_time) min_time = time;
              //std::cout << time << std::endl;
            }
            std::cout << " L-solve: loop = " << loop << std::endl;
            std::cout << "  LOOP_AVG_TIME:  " << ave_time/loop << std::endl;
            std::cout << "  LOOP_MAX_TIME:  " << max_time << std::endl;
            std::cout << "  LOOP_MIN_TIME:  " << min_time << std::endl << std::endl;
          } // end of L-solve

          {
            // ==============================================
            // Preaparing for the first solve with U
            //> create the known solution and set to all 1's ** on host **
            host_scalar_view_t sol_host ("sol_host", nrows);
            //Kokkos::deep_copy (sol_host, ONE);
            Kokkos::Random_XorShift64_Pool<host_execution_space> random(13718);
            Kokkos::fill_random(sol_host, random, scalar_type(1));

            // > create the rhs ** on host **
            // A*sol generates rhs: rhs is dense, use spmv
            host_scalar_view_t rhs_host ("rhs_host", nrows);
            KokkosSparse::spmv ( "N", ONE, U_csr, sol_host, ZERO, rhs_host);

            // ==============================================
            // copy rhs to the default host/device
            scalar_view_t rhs ("rhs", nrows);
            scalar_view_t sol ("sol", nrows);
            Kokkos::deep_copy (rhs, rhs_host);

            // ==============================================
            // do U solve
            timer.reset();
            sptrsv_solve (&khU, sol, rhs);
            Kokkos::fence();
            std::cout << " > Upper-TRI: " << std::endl;
            std::cout << "   Solve Time   : " << timer.seconds() << std::endl;

            // copy solution to host
            Kokkos::deep_copy (sol_host, sol);

            // ==============================================
            // Error Check ** on host **
            Kokkos::fence ();
            std::cout << std::endl;
            if (!check_errors (tol, U_csr, rhs_host, sol_host)) {
              num_failed ++;
            }

            // Benchmark
            // U-solve
            double min_time = 1.0e32;
            double max_time = 0.0;
            double ave_time = 0.0;
            Kokkos::fence ();
            for(int i = 0; i < loop; i++) {
              timer.reset();
              sptrsv_solve (&khU, sol, rhs);
              Kokkos::fence();
              double time = timer.seconds ();
              ave_time += time;
              if(time > max_time) max_time = time;
              if(time < min_time) min_time = time;
              //std::cout << time << std::endl;
            }
            std::cout << " U-solve: loop = " << loop << std::endl;
            std::cout << "  LOOP_AVG_TIME:  " << ave_time/loop << std::endl;
            std::cout << "  LOOP_MAX_TIME:  " << max_time << std::endl;
            std::cout << "  LOOP_MIN_TIME:  " << min_time << std::endl << std::endl;
          } // end of L-solve
        }
        break;
        case CUSPARSE:
        {
          printf("CUSPARSE WRAPPER\n");
          khL.create_sptrsv_handle(SPTRSVAlgorithm::SPTRSV_CUSPARSE, nrows, true);

          using graph_t = typename crsmat_t::StaticCrsGraphType;
          using row_map_t = typename graph_t::row_map_type::non_const_type;
          using entries_t = typename graph_t::entries_type::non_const_type;
          using  values_t = typename crsmat_t::values_type::non_const_type;

          row_map_t row_mapL("row_ptr", L_csr.graph.row_map.extent(0));
          entries_t entriesL("col_idx", L_csr.graph.entries.extent(0));
          values_t   valuesL("values",  L_csr.values.extent(0));

          Kokkos::deep_copy(row_mapL, L_csr.graph.row_map);
          Kokkos::deep_copy(entriesL, L_csr.graph.entries);
          Kokkos::deep_copy( valuesL, L_csr.values);


          sptrsv_symbolic( &khL, row_mapL, entriesL, valuesL );
          {
            // ==============================================
            // Preaparing for the first solve with L
            //> create the known solution and set to all 1's ** on host **
            host_scalar_view_t sol_host ("sol_host", nrows);
            Kokkos::deep_copy (sol_host, ONE);
            //Kokkos::Random_XorShift64_Pool<host_execution_space> random(13718);
            //Kokkos::fill_random(sol_host, random, scalar_type(1));

            // > create the rhs ** on host **
            // A*sol generates rhs: rhs is dense, use spmv
            host_scalar_view_t rhs_host ("rhs_host", nrows);
            KokkosSparse::spmv ( "N", ONE, L_csr, sol_host, ZERO, rhs_host);

            // ==============================================
            // copy rhs to the default host/device
            scalar_view_t rhs ("rhs", nrows);
            scalar_view_t sol ("sol", nrows);
            Kokkos::deep_copy (rhs, rhs_host);

            // ==============================================
            // do L solve
            timer.reset();
            sptrsv_solve (&khL, row_mapL, entriesL, valuesL, rhs, sol);
            Kokkos::fence();
            std::cout << " > Lower-TRI: " << std::endl;
            std::cout << "   Solve Time   : " << timer.seconds() << std::endl;

            // copy solution to host
            Kokkos::deep_copy (sol_host, sol);

            // ==============================================
            // Error Check ** on host **
            Kokkos::fence ();
            std::cout << std::endl;
            if (!check_errors (tol, L_csr, rhs_host, sol_host)) {
              num_failed ++;
            }

            // Benchmark
            // L-solve
            double min_time = 1.0e32;
            double max_time = 0.0;
            double ave_time = 0.0;
            Kokkos::fence ();
            for(int i = 0; i < loop; i++) {
              timer.reset();
              sptrsv_solve (&khL, row_mapL, entriesL, valuesL, rhs, sol);
              Kokkos::fence();
              double time = timer.seconds ();
              ave_time += time;
              if(time > max_time) max_time = time;
              if(time < min_time) min_time = time;
              //std::cout << time << std::endl;
            }
            std::cout << " L-solve: loop = " << loop << std::endl;
            std::cout << "  LOOP_AVG_TIME:  " << ave_time/loop << std::endl;
            std::cout << "  LOOP_MAX_TIME:  " << max_time << std::endl;
            std::cout << "  LOOP_MIN_TIME:  " << min_time << std::endl << std::endl;
          }

          //khU.create_sptrsv_handle(SPTRSVAlgorithm::SPTRSV_CUSPARSE, nrows, false);
          //sptrsv_symbolic( &khU, U_csr.graph.row_map, U_csr.graph.entries, U_csr.values );
        }
        break;
        default:
          std::cout << " > Invalid test ID < " << std::endl;
          exit (0);
      }
    }
  }
  std::cout << std::endl << std::endl;

  return num_failed;
}


void print_help_sptrsv() {
  printf("Options:\n");
  printf("  --test [OPTION] : Use different kernel implementations\n");
  printf("                    Options:\n");
  printf("                    superlu-naive, superlu-dag, superlu-spmv-dag\n\n");
  printf("  -f [file]       : Read in matrix in Matrix Market formatted text file 'file'.\n");
  printf("  --loop [LOOP]   : How many spmv to run to aggregate average time. \n");
}


int main(int argc, char **argv) {
  std::vector<int> tests;
  std::string matrix_filename;

  int loop = 1;
  // parameter for ILU(k)
  int expand_fact = 6;
  // merge supernodes
  bool merge = false;
  // invert diagonal of L-factor
  bool invert_diag = false;
  // invert off-diagonal of L-factor
  bool invert_offdiag = false;
  // store U in CSR, or CSC
  bool u_in_csr = true;
  // use metis
  bool metis = false;
  // verbose
  bool verbose = true;

  if(argc == 1)
  {
    print_help_sptrsv();
    return 0;
  }

  for(int i = 0; i < argc; i++) {
    if((strcmp(argv[i],"--test")==0)) {
      i++;
      if((strcmp(argv[i],"superlu-naive")==0)) {
        tests.push_back( SUPERNODAL_NAIVE );
      }
      if((strcmp(argv[i],"superlu-dag")==0)) {
        tests.push_back( SUPERNODAL_DAG );
      }
      if((strcmp(argv[i],"superlu-spmv-dag")==0)) {
        tests.push_back( SUPERNODAL_SPMV_DAG );
      }
      if((strcmp(argv[i],"cusparse")==0)) {
        tests.push_back( CUSPARSE );
      }
      continue;
    }
    if((strcmp(argv[i],"-f")==0)) {
      matrix_filename = argv[++i];
      continue;
    }
    if((strcmp(argv[i],"--quiet")==0)) {
      verbose = false;
      continue;
    }
    if((strcmp(argv[i],"--expand-fact")==0)) {
      expand_fact = atoi(argv[++i]);
      continue;
    }
    if((strcmp(argv[i],"--loop")==0)) {
      loop = atoi(argv[++i]);
      continue;
    }
    /* not supported through this interface, yet */
    if((strcmp(argv[i],"--merge")==0)) {
      merge = true;
      continue;
    }
    if((strcmp(argv[i],"--invert-diag")==0)) {
      invert_diag = true;
      continue;
    }
    if((strcmp(argv[i],"--invert-offdiag")==0)) {
      invert_offdiag = true;
      continue;
    }
    if((strcmp(argv[i],"--u-in-csc")==0)) {
      u_in_csr = false;
      continue;
    }
    if((strcmp(argv[i],"--metis")==0)) {
      metis = true;
      continue;
    }
    if((strcmp(argv[i],"--help")==0) || (strcmp(argv[i],"-h")==0)) {
      print_help_sptrsv();
      return 0;
    }
  }

  std::cout << std::endl;
  for (size_t i = 0; i < tests.size(); ++i) {
    std::cout << "tests[" << i << "] = " << tests[i] << std::endl;
  }

  {
    using scalar_t = double;
    //using scalar_t = Kokkos::complex<double>;
    Kokkos::ScopeGuard kokkosScope (argc, argv);
    int total_errors = test_sptrsv_perf<scalar_t> (tests, verbose, matrix_filename,
                                                   metis, expand_fact, merge, invert_diag, invert_offdiag, u_in_csr, loop);
    if(total_errors == 0)
      std::cout << "Kokkos::SPTRSV Test: Passed"
                << std::endl << std::endl;
    else
      std::cout << "Kokkos::SPTRSV Test: Failed (" << total_errors << " / " << 2*tests.size() << " failed)"
                << std::endl << std::endl;
  }
  return 0;
}
#else // defined(KOKKOSKERNELS_ENABLE_SUPERNODAL_SPTRSV)
int main(int argc, char **argv) {
  std::cout << std::endl << " ** SUPERNODAL NOT ENABLED **" << std::endl << std::endl;
  exit(0);
  return 0;
}
#endif

#else // defined( KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA ) && (!defined(KOKKOS_ENABLE_CUDA) || ( 8000 <= CUDA_VERSION ))

int main(int argc, char **argv) {
#if !defined(KOKKOSKERNELS_INST_DOUBLE)
  std::cout << " Only supported with double precision" << std::endl;
#endif
#if !defined( KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA )
  std::cout << " KOKKOS_ENABLE_CXX11_DISPATCH_LAMBDA **not** defined" << std::endl;
#endif
#if defined(KOKKOS_ENABLE_CUDA)
  std::cout << " KOKKOS_ENABLE_CUDA defined" << std::endl;
  #if !defined(KOKKOS_ENABLE_CUDA_LAMBDA)
  std::cout << " KOKKOS_ENABLE_CUDA_LAMBDA **not** defined" << std::endl;
  #endif
  std::cout << " CUDA_VERSION = " << CUDA_VERSION << std::endl;
#endif
  return 0;
}
#endif
