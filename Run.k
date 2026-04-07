$# LS-DYNA Keyword file created by LS-PrePost(R) XXXX RX (vX.XX.X)
$# Created on XXX-XX-XXXX (XX:XX:XX)
*KEYWORD
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
*TITLE
$#                                                                         title
HVI_EnKF_sample
*MAT_JOHNSON_COOK
*EOS_LINEAR_POLYNOMIAL
*EOS_GRUNEISEN
*ELEMENT_SPH
*NODE
*SET_NODE_LIST
*PART
*SECTION_SPH
*INITIAL_VELOCITY_GENERATION
*CONTROL_CONTACT
*CONTROL_ENERGY
*CONTROL_SOLUTION
*CONTROL_SPH
*CONTROL_TERMINATION
*CONTROL_TIMESTEP
*DATABASE_GLSTAT
*DATABASE_MATSUM
*DATABASE_NODOUT
*DATABASE_SPHOUT
$*DATABASE_NODOUT
*DATABASE_BINARY_D3PLOT
*DATABASE_EXTENT_BINARY
*DATABASE_HISTORY_NODE_SET
*END
