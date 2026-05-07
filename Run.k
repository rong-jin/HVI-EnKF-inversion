$# LS-DYNA Keyword file created by LS-PrePost(R) XXXX RX (vX.XX.X)
$# Created on XXX-XX-XXXX (XX:XX:XX)
$===============================================================================
$ Unit system used in this LS-DYNA model
$===============================================================================
$ This model uses a consistent g-cm-s unit system.
$
$ Base units:
$   Mass        = g
$   Length      = cm
$   Time        = s
$
$ Derived units:
$   Force       = dyne      = g*cm/s^2
$   Stress      = dyne/cm^2
$   Energy      = erg       = dyne*cm
$
$ Example values in this unit system:
$   Density     = 7.83E+00   g/cm^3
$   Young's E   = 2.07E+12   dyne/cm^2
$   Velocity    = 1.56E+03   cm/s
$   Gravity     = 9.806E+02  cm/s^2
$
$ All material parameters, EOS parameters, loads, velocities, and database outputs
$ should be interpreted consistently in the g-cm-s unit system.
$===============================================================================
$===============================================================================
*KEYWORD
$===============================================================================
*TITLE
$#                                                                         title
HVI-EnKF
*PARAMETER
$#   prmr1      val1     prmr2      val2     prmr3      val3     prmr4      val4
RA1,2251710e3                                                           
RB1,1683460e3                                                              
Rn1,0.242                                                                 
RC1,0.013                                                                 
Rm1,1.550                                                                 
RD11,-0.35                                                               
RD21,0.6025                                                                
RD31,-0.4537                                                               
RD41,0.4738                                                                
RD51,7.2 
RCS,452000 
RS1,1.242 
RG,1.54 
RA0,0.33 
RT,1.2E-5                                                                  
$===============================================================================
*CONTROL_CONTACT
$#  slsfac    rwpnal    islchk    shlthk    penopt    thkchg     orien    enmass
       0.0       0.0         0         0         0         0         0         0
$#  usrstr    usrfrc     nsbcs    interm     xpene     ssthk      ecdt   tiedprj
         0         0         0         0       0.0         0         0         0
$#   sfric     dfric       edc       vfc        th     th_sf    pen_sf     ptscl
       0.0       0.0       0.0       0.0       0.0       0.0       0.0       1.0
$#  ignore    frceng   skiprwg    outseg   spotstp   spotdel   spothin   dir_tie
         0         0         0         0         0         0       0.0         0
$#    isym    nserod    rwgaps    rwgdth     rwksf      icov    swradf    ithoff
         0         0         1       0.0       0.0         0       0.0         0
$#  shledg    pstiff    ithcnt    tdcnof     ftall    unused    shltrw    igactc
         0         0         0         0         0                 0.0         0
$===============================================================================
*CONTROL_ENERGY
$#    hgen      rwen    slnten     rylen     irgen     maten     drlen     disen
         0         0         0         0         0         0         0         0
$===============================================================================
*CONTROL_SOLUTION
$#    soln       nlq     isnan     lcint     lcacc     ncdcf    nocopy      crvp
         0         0         0         0         0         0         0         0
$===============================================================================
*CONTROL_SPH
$#    ncbs     boxid        dt      idim   nmneigh      form     start      maxv
         1         0    1.0E20         3       150         0       0.0    1.0E15
$===============================================================================
*CONTROL_TERMINATION
$#  endtim    endcyc     dtmin    endeng    endmas     nosol
&t                 0       0.0       0.0       0.0         0
$===============================================================================
$ OUTPUT
$===============================================================================
*CONTROL_TIMESTEP
$#  dtinit    tssfac      isdo    tslimt     dt2ms      lctm     erode     ms1st
1.0000E-12       0.0         0       0.0       0.0         0         0         0
*DATABASE_GLSTAT
$#      dt    binary      lcur     ioopt
1.00000E-7         1         0         1
*DATABASE_MATSUM
$#      dt    binary      lcur     ioopt
1.00000E-7         1         0         1
*DATABASE_NODOUT
$#      dt    binary      lcur     ioopt   option1   option2
1.00000E-7         1         0         1       0.0         0
*DATABASE_BINARY_D3PLOT
$#      dt      lcdt      beam     npltc    psetid
1.00000E-7         0         0         0         0
$===============================================================================
*DATABASE_EXTENT_BINARY
$#   neiph     neips    maxint    strflg    sigflg    epsflg    rltflg    engflg
         0         0         0         1         1         1         1         1
$#  cmpflg    ieverp    beamip     dcomp      shge     stssz    n3thdt   ialemat
         0         0         0         1         1         1         2         0
$# nintsld   pkp_sen      sclp     hydro     msscl     therm    intout    nodout
         0         0       0.0         0         0         1                    
$#    dtdt    resplt     neipb   quadsld    cubsld   deleres
         0         0         0         0         0         0
$===============================================================================
*DATABASE_HISTORY_NODE_SET
$#     id1       id2       id3       id4       id5       id6       id7       id8
         2         0         0         0         0         0         0         0
*MAT_JOHNSON_COOK_TITLE
$#     mid        ro         g         e        pr       dtf        vp    rateop
         2        0          0         0       0.0       0.0       0.0       0.0
$#       a         b         n         c         m        tm        tr      epso
       &a1       &b1       &n1       &c1       &m1     905.0     298.0     0.001
$#      cp        pc     spall        it        d1        d2        d3        d4
         0       0.0       0.0       0.0      &d11      &d21      &d31      &d41      
$#      d5c2/p/xnp/d      erod     efmin    numint         k      eps1
&d51             0.0       0.0       0.0       0.0       0.0       0.0
*MAT_JOHNSON_COOK_TITLE
$#     id1       id2       id3       id4       id5       id6       id7       id8
         1         0         0         0         0         0         0         0
$#       a         b         n         c         m        tm        tr      epso
       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0
$#      cp        pc     spall        it        d1        d2        d3        d4
       0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0
$#      d5c2/p/xnp/d      erod     efmin    numint         k      eps1
       0.0       0.0       0.0       0.0       0.0       0.0       0.0
$===============================================================================
$ INITIAL / CONTROL SETTINGS
$===============================================================================
*MAT_THERMAL_ISOTROPIC
$  TMID  TRO  TGRLC  TGMULT  TLAT  HLAT
$#    tmid        ro     tgrlc    tgmult      tlat      hlat
         1       0.0       0.0       0.0       0.0       0.0
$  HC  TC
$#      hc        tc
       0.0         0
$===============================================================================
*INITIAL_TEMPERATURE_SET
$#    nsid      temp       loc
         0    298.15         0
$===============================================================================
*INITIAL_VELOCITY_GENERATION
$#      id      styp     omega        vx        vy        vz     ivatn      icid
         0       0.0       0.0       0.0       0.0       0.0       0.0       0.0
$#      xc        yc        zc        nx        ny        nz     phase    irigid
       0.0       0.0       0.0       0.0       0.0       0.0         0         0
*EOS_LINEAR_POLYNOMIAL
$#   eosid        c0        c1        c2        c3        c4        c5        c6
         1       0.0       0.0       0.0       0.0       0.0       0.0       0.0
$#      e0        v0
       0.0       1.0
*EOS_GRUNEISEN
$#   eosid         c        s1        s2        s3    gamma0         a        e0
         2       &cs       &s1       0.0       0.0        &g       &a0      0.0
$V0
$#      v0    unused      lcid
       1.0                   0
$===============================================================================
*ELEMENT_SPH
$#   nid     pid            mass      nend
       1       1             0.0         0
$===============================================================================
$ GEOMETRY / DISCRETIZATION
$===============================================================================
*NODE
$#   nid               x               y               z      tc      rc
       1               0               0               0       0       0
$===============================================================================
*PART
$# title
Projectile
$#     pid     secid       mid     eosid      hgid      grav    adpopt      tmid
         1         1         1         1         0         0         0         0
*SECTION_SPH_TITLE
sph
$#   secid      cslh      hmin      hmax    sphini     death     start   sphkern
         1       1.2       0.2       2.0       0.0    1.0E20       0.0         0
*PART
$#                                                                         title
Target
$#     pid     secid       mid     eosid      hgid      grav    adpopt      tmid
   5000001         1         2         2         0         0         0         1
$===============================================================================
*SET_NODE_LIST
$#     sid       da1       da2       da3       da4    solver       its         -
         1       0.0       0.0       0.0       0.0MECH               1          
$#    nid1      nid2      nid3      nid4      nid5      nid6      nid7      nid8
*END

