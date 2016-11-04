Gunrock v0.4 Release Notes {#release_notes}
==========================

Release 0.4
?th October 2015

Gunrock release 0.4 is a feature release that adds 
 - New optimizations to both advance and filter operators,
 - Multi-iteration supports for BFS, SSSP, BC, CC and PR,
 - Better error handling,
 - Updates on several interfaces,
 - Overall performance improvement for both single and multi-GPU execution.

v0.4 ChangeLog
==============
 - Integrated direction-optimizing BFS with normal BFS.
 - Added three new strategies of advance:
    - ALL_EDGES, optimized for advance on all edges with all 
      vertices of the graph, no need to use sorted search for load balancing, 
      just binary search over the whole row offsets array;
      used in CC.
    - LB_CULL, fuzed LB advance with a subsequent CULL filter;
      used in BFS, SSSP and BC.
    - LB_LIGHT_CULL, fuzed LB_LIGHT advance with a subsequent CULL filter;
      used in BFS, SSSP and BC.
 - Added three new strategies of filter: 
    - COMPACTED_CULL, optimized on several culling heuristics;
    - SIMPLIFIED, an other implementation of the CULL filter, without 
      some optimizations;
    - BY_PASS, optimized for filter with no elements 
      to remove from the input frontier; used in CC and PR.
 - Added multi-iteration support for BFS, SSSP, BC, CC and PR.

v0.4 Known Issues
=================
 - HITS, and SALSA do not have CPU reference yet.
 - HITS, SALSA, Who-to-Follow do not have multi-GPU support yet.
 - Out of memory error will cause result validation to fail, for graphs that 
   approching the memory limit of GPUs.