# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:38:01 2018

@author: Sundukov
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
import ctypes
import logging
import ThermoLog
import copy
import CoolProp.CoolProp as CP
# from scipy.optimize import root 
# from scipy.optimize import minimize

# ThermoLog.setup_logger('solverLog', 'info.log',logging.DEBUG)
solverLog=logging.getLogger('solverLog')
# solverLog.propagate = False

"""
тест
Много данных есть в NASA/TP—2002-211556 и в Third Millennium Ideal Gas and Condensed Phase
Thermochemical Database for Combustion with Updates from Active Thermochemical Tables

Исходные данные по некоторым веществам из NASA Glenn Coefficients for Calculating
Thermodynamic Properties of Individual Species (NASA/TP—2002-211556)

Air Mole%:N2 78.084,O2 20.9476,Ar .9365,CO2 .0319.Gordon,1982.Reac
2 g 9/95 N 1.5617O .41959AR.00937C .00032 .00000 0 28.9651159 -125.530
200.000 1000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 8649.264
1.009950160D+04-1.968275610D+02 5.009155110D+00-5.761013730D-03 1.066859930D-05
-7.940297970D-09 2.185231910D-12 -1.767967310D+02-3.921504225D+00
1000.000 6000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 8649.264
2.415214430D+05-1.257874600D+03 5.144558670D+00-2.138541790D-04 7.065227840D-08
-1.071483490D-11 6.577800150D-16 6.462263190D+03-8.147411905D+00

Ar Ref-Elm. Moore,1971. Gordon,1999.
3 g 3/98 AR 1.00 0.00 0.00 0.00 0.00 0 39.9480000 0.000
200.000 1000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 6197.428
0.000000000D+00 0.000000000D+00 2.500000000D+00 0.000000000D+00 0.000000000D+00
0.000000000D+00 0.000000000D+00 -7.453750000D+02 4.379674910D+00
1000.000 6000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 6197.428
2.010538475D+01-5.992661070D-02 2.500069401D+00-3.992141160D-08 1.205272140D-11
-1.819015576D-15 1.078576636D-19 -7.449939610D+02 4.379180110D+00
6000.000 20000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 6197.428
-9.951265080D+08 6.458887260D+05-1.675894697D+02 2.319933363D-02-1.721080911D-06
6.531938460D-11-9.740147729D-16 -5.078300340D+06 1.465298484D+03

CO2 Gurvich,1991 pt1 p27 pt2 p24.
3 g 9/99 C 1.00O 2.00 0.00 0.00 0.00 0 44.0095000 -393510.000
200.000 1000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 9365.469
4.943650540D+04-6.264116010D+02 5.301725240D+00 2.503813816D-03-2.127308728D-07
-7.689988780D-10 2.849677801D-13 -4.528198460D+04-7.048279440D+00
1000.000 6000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 9365.469
1.176962419D+05-1.788791477D+03 8.291523190D+00-9.223156780D-05 4.863676880D-09
-1.891053312D-12 6.330036590D-16 -3.908350590D+04-2.652669281D+01
6000.000 20000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 9365.469
-1.544423287D+09 1.016847056D+06-2.561405230D+02 3.369401080D-02-2.181184337D-06
6.991420840D-11-8.842351500D-16 -8.043214510D+06 2.254177493D+03

O2 Ref-Elm. Gurvich,1989 pt1 p94 pt2 p9.
3 tpis89 O 2.00 0.00 0.00 0.00 0.00 0 31.9988000 0.000
200.000 1000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 8680.104
-3.425563420D+04 4.847000970D+02 1.119010961D+00 4.293889240D-03-6.836300520D-07
-2.023372700D-09 1.039040018D-12 -3.391454870D+03 1.849699470D+01
1000.000 6000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 8680.104
-1.037939022D+06 2.344830282D+03 1.819732036D+00 1.267847582D-03-2.188067988D-07
2.053719572D-11-8.193467050D-16 -1.689010929D+04 1.738716506D+01
6000.000 20000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 8680.104
4.975294300D+08-2.866106874D+05 6.690352250D+01-6.169959020D-03 3.016396027D-07
-7.421416600D-12 7.278175770D-17 2.293554027D+06-5.530621610D+02

N2 Ref-Elm. Gurvich,1978 pt1 p280 pt2 p207.
3 tpis78 N 2.00 0.00 0.00 0.00 0.00 0 28.0134000 0.000
200.000 1000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 8670.104
2.210371497D+04-3.818461820D+02 6.082738360D+00-8.530914410D-03 1.384646189D-05
-9.625793620D-09 2.519705809D-12 7.108460860D+02-1.076003744D+01
1000.000 6000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 8670.104
5.877124060D+05-2.239249073D+03 6.066949220D+00-6.139685500D-04 1.491806679D-07
-1.923105485D-11 1.061954386D-15 1.283210415D+04-1.586640027D+01
6000.000 20000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 8670.104
8.310139160D+08-6.420733540D+05 2.020264635D+02-3.065092046D-02 2.486903333D-06
-9.705954110D-11 1.437538881D-15 4.938707040D+06-1.672099740D+03

H2O Hf:Cox,1989. Woolley,1987. TRC(10/88) tuv25.
2 g 8/89 H 2.00O 1.00 0.00 0.00 0.00 0 18.0152800 -241826.000
200.000 1000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 9904.092
-3.947960830D+04 5.755731020D+02 9.317826530D-01 7.222712860D-03-7.342557370D-06
4.955043490D-09-1.336933246D-12 -3.303974310D+04 1.724205775D+01
1000.000 6000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 9904.092
1.034972096D+06-2.412698562D+03 4.646110780D+00 2.291998307D-03-6.836830480D-07
9.426468930D-11-4.822380530D-15 -1.384286509D+04-7.978148510D+00

CH4 Gurvich,1991 pt1 p44 pt2 p36.
2 g 8/99 C 1.00H 4.00 0.00 0.00 0.00 0 16.0424600 -74600.000
200.000 1000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 10016.202
-1.766850998D+05 2.786181020D+03-1.202577850D+01 3.917619290D-02-3.619054430D-05
2.026853043D-08-4.976705490D-12 -2.331314360D+04 8.904322750D+01
1000.000 6000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 10016.202
3.730042760D+06-1.383501485D+04 2.049107091D+01-1.961974759D-03 4.727313040D-07
-3.728814690D-11 1.623737207D-15 7.532066910D+04-1.219124889D+02

CH3 D0(H3C-H): Ruscic,1999. Jacox,1998.
2 g 4/02 C 1.00H 3.00 0.00 0.00 0.00 0 15.0345200 146658.040
200.000 1000.000 7 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 10366.340
-2.876188806D+04 5.093268660D+02 2.002143949D-01 1.363605829D-02-1.433989346D-05
1.013556725D-08-3.027331936D-12 0.000000000D+00 1.408271825D+04 2.022772791D+01
1000.000 6000.000 7 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 10366.340
2.760802663D+06-9.336531170D+03 1.487729606D+01-1.439429774D-03 2.444477951D-07
-2.224555778D-11 8.395065760D-16 0.000000000D+00 7.481809480D+04-7.919682400D+01

Данные по западным топливам:
JP-4 McBride,1996 pp85,93. Hcomb = 18640.BTU/#
0 g 6/96 C 1.00H 1.94 0.00 0.00 0.00 1 13.9661036 -22723.000
298.150 0.0000 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.000
JP-5 Or ASTMA1(L). McBride,1996 pp85,93. Hcomb = 18600.BTU/#
0 g 6/96 C 1.00H 1.92 0.00 0.00 0.00 1 13.9459448 -22183.000
298.150 0.0000 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.000
JP-10(L) Exo-tetrahydrodicyclopentadiene. Smith,1979. React.
0 g 6/01 C 10.00H 16.00 0.00 0.00 0.00 0 136.2340400 -122800.400
298.150 0.0000 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.000
JP-10(g) Exo-tetrahydrodicyclopentadiene.Pri.Com.R.Jaffe 12/00. React.
2 g 6/01 C 10.00H 16.00 0.00 0.00 0.00 0 136.2340400 -86855.900
200.000 1000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 22997.434
-7.310769440D+05 1.521764245D+04-1.139312644D+02 4.281501620D-01-5.218740440D-04
3.357233400D-07-8.805750980D-11 -8.067482120D+04 6.320148610D+02
1000.000 6000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 22997.434
1.220329594D+07-5.794846240D+04 1.092281156D+02-1.082406215D-02 2.034992622D-06
-2.052060369D-10 8.575760210D-15 3.257334050D+05-7.092350760D+02
Jet-A(L) McBride,1996. Faith,1971. Gracia-Salcedo,1988. React.
1 g 2/96 C 12.00H 23.00 0.00 0.00 0.00 1 167.3110200 -303403.000
220.000 550.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 0.000
-4.218262130D+05-5.576600450D+03 1.522120958D+02-8.610197550D-01 3.071662234D-03
-4.702789540D-06 2.743019833D-09 -3.238369150D+04-6.781094910D+02
Jet-A(g) McBride,1996. Faith,1971. Gracia-Salcedo,1988. React.
2 g 8/01 C 12.00H 23.00 0.00 0.00 0.00 0 167.3110200 -249657.000
273.150 1000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 0.000
-6.068695590D+05 8.328259590D+03-4.312321270D+01 2.572390455D-01-2.629316040D-04
1.644988940D-07-4.645335140D-11 -7.606962760D+04 2.794305937D+02
1000.000 6000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 0.000
1.858356102D+07-7.677219890D+04 1.419826133D+02-7.437524530D-03 5.856202550D-07
1.223955647D-11-3.149201922D-15 4.221989520D+05-8.986061040D+02

Данные по продуктам сгорания воздуха и керосина из Gas turbine Performance (стр115):
_Cp=A0+A1*Tz+A2*Tz^2+A3*Tz^3+A4*Tz^4+A5*Tz^5+A6*Tz^6+A7*Tz^7+A8*Tz^8+FAR/(1+FAR)*
*(B0+B1*Tz+B2*Tz^2+B3*Tz^3+B4*Tz^4+B5*Tz^5+B6*Tz^6+B7*Tz^7)
H =A0*TZ + A1/2*TZ^2 + A2/3*TZ^3 + A3/4*TZ^4 + A4/5*TZ^5 + A5/6*TZ^6 + A6/7*TZ^7 +
+ A7/8*TZ^8 + A8/9*TZ^9 + A9 + (FAR/(1+FAR))*(B0*TZ + B1/2*TZ^2 + B2/3*TZ^3 +
+ B3/4*TZ^4 + B4/5*TZ^5 + B5/6*TZ^6 + B6/7*TZ^7 + B8)
Where TZ = TS/1000 
FAR=DeltaH/(LowHeatValue*EfficiencyCombustion)
Entropy: CP/TdT = FT2 - FT1
FT2 = A0*ln(T2Z) + A1*T2Z + A2/2*T2Z^2 + A3/3*T2Z^3 + A4/4*T2Z^4 + A5/5*T2Z^5 +
+ A6/6*T2Z^6 + A7/7*T2Z^7 + A8/8*T2Z^8 + A10 + (FAR/(1 + FAR)) * (B0*ln(T2) +
+ B1*TZ + B2/2*TZ^2 + B3/3*TZ^3 + B4/4*TZ^4 + B5/5*TZ^5 + B6/6*TZ^6 + B7/7*TZ^7 + B9)
FT1 = A0*ln(T1Z) + A1*T1Z + A2/2*T1Z^2 + A3/3*T1Z^3 + A4/4*T1Z^4 + A5/5*T1Z^5 +
+ A6/6*T1Z^6 + A7/7*T1Z^7 + A8/8*T1Z^8 + A10 + (FAR/(1+FAR))*(B0*ln(T1) + B1*TZ +
+ B2/2*TZ^2 + B3/3*TZ^3 + B4/4*TZ^4 + B5/5*TZ^5 + B6/6*TZ^6 + B7/7*TZ^7 + B9)
Where T2Z = TS2/1000, T1Z = TS1/1000 (в формулах FT2 и FT1 что-то напутано  стемпературами, скорее всего в FT2 везде используется T2Z, а в FT1 - T1Z)
A0=0.992313 A1=0.236688 A2=-1.852148 A3=6.083152 A4=-8.893933 A5=7.097112 A6=-3.234725
A7=0.794571 A8=-0.081873 A9=0.422178 A10=0.001053
B0= -0.718874, B1=8.747481, B2= -15.863157, B3=17.254096, B4= -10.233795,
B5=3.081778, B6= -0.361112, B7= -0.003919, B8=0.0555930, B9= -0.0016079.

Kerosene - average composition of C12H23.5 and molecular weight of 167.7 (gas turbine performance - стр588)

характеристики топлива для коммерческой авиации на западе (вероятно ТС-1)
Jet-A(L) McBride,1996. Faith,1971. Gracia-Salcedo,1988. React.
1 g 2/96 C 12.00H 23.00 0.00 0.00 0.00 1 167.3110200 -303403.000
220.000 550.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 0.000
-4.218262130D+05-5.576600450D+03 1.522120958D+02-8.610197550D-01 3.071662234D-03
-4.702789540D-06 2.743019833D-09 -3.238369150D+04-6.781094910D+02
Jet-A(g) McBride,1996. Faith,1971. Gracia-Salcedo,1988. React.
2 g 8/01 C 12.00H 23.00 0.00 0.00 0.00 0 167.3110200 -249657.000
273.150 1000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 0.000
-6.068695590D+05 8.328259590D+03-4.312321270D+01 2.572390455D-01-2.629316040D-04
1.644988940D-07-4.645335140D-11 -7.606962760D+04 2.794305937D+02
1000.000 6000.0007 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 0.0 0.000
1.858356102D+07-7.677219890D+04 1.419826133D+02-7.437524530D-03 5.856202550D-07
1.223955647D-11-3.149201922D-15 4.221989520D+05-8.986061040D+02
"""

#коэффициенты для расчета свойств воздуха по данным НАСА TP-2002-211556 в диапазоне от 200 до 1000К Mole%:N2 78.084,O2 20.9476,Ar .9365,CO2 .0319.Gordon,1982.Reac 
#coefNASA=np.array([1.009950160E+04,-1.968275610E+02,5.009155110E+00,-5.761013730E-03,1.066859930E-05,-7.940297970E-09,2.185231910E-12,-1.767967310E+02,-3.921504225E+00])

#СОСТАВ ВОЗДУХА - ТУТ МНОГО ИНФЫ: https://en.wikipedia.org/wiki/Template:Table_composition_of_dry_atmosphere будем использовать данные ICAO, т.к. мы ориентируемся на двигатели для авиации
#состав воздуха в объемынх долях тогда: N2=78.084% O2=20.9476% Ar=0.9340% CO2=0.0314% (Ne=0.001818% He=0.000524%)
#mole_comp_ICAO=(0.78084, 0.209476, 0.00934, 0.000314)


#ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ
Tzero=200 #Кельвин, условно нулевая температура относительно которой начинается отсчет абсолютного значения энтропии S в одноименной функции
Pzero=101325 #Паскаль, условно нулевое давление относительно которого начинается отсчет абсолютного значения энтропии S в одноименной функции


#состав рабочего тела задаем в формате mass_comp(N2,O2,Ar,CO2,H2O,керосин_газ,керосин_жидкий,метан)
#если в составе будем меняться индекс расположения воды, то нужно откорректировать функцию  WAR_to_moist_air

Runiversal=8.314510 #J/(mol-K) NASA
MolW_Air=28.9651159e-3 #kg/mol NASA
MolW_N2=28.0134000e-3#kg/mol 
MolW_O2=31.9988000e-3#kg/mol 
MolW_CO2=44.0095000e-3#kg/mol 
MolW_Ar=39.9480000e-3#kg/mol 
MolW_H2O=18.0152800e-3#kg/mol 
MolW_JetA=167.3110200e-3#kg/mol
MolW_CH4=16.0424600e-3#kg/mol
MolW_CH3=15.0345200e-3#kg/mol
MolW_dict={'N2':MolW_N2,
           'O2':MolW_O2,
           'CO2':MolW_CO2,
           'Ar':MolW_Ar,
           'H2O':MolW_H2O,
           'JetA_gas':MolW_JetA,
           'JetA_liquid':MolW_JetA,
           'CH4':MolW_CH4,
           'CH3':MolW_CH3,
           'Air':MolW_Air}
#задаем список со значениями всех молярных масс веществ, используемых в расчете:
# MolW_v=np.array((MolW_N2,MolW_O2,MolW_Ar,MolW_CO2,MolW_H2O,MolW_JetA,MolW_JetA))

#критические температуры веществ для вычисления вязкости по методике из CFX solver guide - раздел 1.4.7.1 стр.44 Interacting Sphere Models "chung1984 - Applications of kinetic gas theories and multiparameter correlation for prediction of dilute gas viscosity and thermal conductivity"
T_critical_N2=126.2 #https://webbook.nist.gov/cgi/cbook.cgi?ID=C7727379&Mask=4
T_critical_O2=154.58 #https://webbook.nist.gov/cgi/inchi?ID=C7782447&Mask=4#Thermo-Phase
T_critical_CO2=304.18 #https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Units=SI&Mask=4#Thermo-Phase
T_critical_Ar=150.86 #https://webbook.nist.gov/cgi/cbook.cgi?ID=C7440371&Units=SI&Mask=4#Thermo-Phase
T_critical_H2O=647.1 #https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Units=SI&Mask=4#Thermo-Phase
T_critical_JetA=661.15 #TODO! неточные данные нужно проверить!  по книге "Энергоемкие горючие для авиационных и ракетных двигателей", Л.С.Яновский, Москва, 2009, стр19, Тс=660К
T_critical_CH4=190.6 #https://webbook.nist.gov/cgi/cbook.cgi?ID=C74828&Units=SI&Mask=4#Thermo-Phase
T_critical_Air=132.5306 #coolprop air
T_critical_dict={'N2':T_critical_N2,
           'O2':T_critical_O2,
           'CO2':T_critical_CO2,
           'Ar':T_critical_Ar,
           'H2O':T_critical_H2O,
           'JetA_gas':T_critical_JetA,
           'JetA_liquid':T_critical_JetA,
           'CH4':T_critical_CH4,
            'Air':T_critical_Air}

#критические плотности веществ для вычисления вязкости по методике из CFX solver guide - раздел 1.4.7.1 стр.44 Interacting Sphere Models "chung1984 - Applications of kinetic gas theories and multiparameter correlation for prediction of dilute gas viscosity and thermal conductivity"
Ro_critical_N2=313.189812 #kg/m3, 11.18mol/litre https://webbook.nist.gov/cgi/cbook.cgi?ID=C7727379&Mask=4
Ro_critical_O2=435.18368 #kg/m3, 13.60mol/litre https://webbook.nist.gov/cgi/inchi?ID=C7782447&Mask=4#Thermo-Phase
Ro_critical_CO2=466.060605 #kg/m3, 10.59mol/litre https://webbook.nist.gov/cgi/cbook.cgi?ID=C124389&Units=SI&Mask=4#Thermo-Phase
Ro_critical_Ar=535.70268 #kg/m3, 13.41mol/litre https://webbook.nist.gov/cgi/cbook.cgi?ID=C7440371&Units=SI&Mask=4#Thermo-Phase
Ro_critical_H2O=322.6536648 #kg/m3, 17.91mol/litre https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Units=SI&Mask=4#Thermo-Phase
Ro_critical_JetA=252 #kg/m3 0TODO! неточные данные нужно проверить! по книге "Энергоемкие горючие для авиационных и ракетных двигателей", Л.С.Яновский, Москва, 2009, стр19, Roс=252kg/m3 для керосина Т-1, который очень близкий аналог Jet-A
Ro_critical_CH4=162.028846 #kg/m3, 10.1mol/litre https://webbook.nist.gov/cgi/cbook.cgi?ID=C74828&Units=SI&Mask=4#Thermo-Phase
Ro_critical_Air=342.68456416799995 #kg/m3 coolprop air
Ro_critical_dict={'N2':Ro_critical_N2,
           'O2':Ro_critical_O2,
           'CO2':Ro_critical_CO2,
           'Ar':Ro_critical_Ar,
           'H2O':Ro_critical_H2O,
           'JetA_gas':Ro_critical_JetA,
           'JetA_liquid':Ro_critical_JetA,
           'CH4':Ro_critical_CH4,
            'Air':Ro_critical_Air}

#исходники для расчета свойств веществ: (первый кортеж - коэффициенты для диапазона температур 200-1000К, второй 1000-6000К, третий - теплота образования при температуре 298.15К)
coefsN2=np.array(((2.210371497E+04,-3.818461820E+02, 6.082738360E+00,-8.530914410E-03, 1.384646189E-05,-9.625793620E-09, 2.519705809E-12, 7.108460860E+02,-1.076003744E+01),(5.877124060E+05,-2.239249073E+03, 6.066949220E+00,-6.139685500E-04, 1.491806679E-07,-1.923105485E-11, 1.061954386E-15, 1.283210415E+04,-1.586640027E+01),(0.000)),dtype=object)
coefsO2=np.array(((-3.425563420E+04,4.847000970E+02, 1.119010961E+00, 4.293889240E-03,-6.836300520E-07,-2.023372700E-09, 1.039040018E-12, -3.391454870E+03, 1.849699470E+01),(-1.037939022E+06, 2.344830282E+03, 1.819732036E+00, 1.267847582E-03,-2.188067988E-07,2.053719572E-11,-8.193467050E-16, -1.689010929E+04, 1.738716506E+01),(0.000)),dtype=object)
coefsCO2=np.array(((4.943650540E+04,-6.264116010E+02, 5.301725240E+00, 2.503813816E-03,-2.127308728E-07,-7.689988780E-10, 2.849677801E-13, -4.528198460E+04,-7.048279440E+00),(1.176962419E+05,-1.788791477E+03, 8.291523190E+00,-9.223156780E-05, 4.863676880E-09,-1.891053312E-12, 6.330036590E-16, -3.908350590E+04,-2.652669281E+01),(-393510.000)),dtype=object)
coefsAr=np.array(((0.000000000E+00, 0.000000000E+00, 2.500000000E+00, 0.000000000E+00, 0.000000000E+00,0.000000000E+00, 0.000000000E+00, -7.453750000E+02, 4.379674910E+00),(2.010538475E+01,-5.992661070E-02, 2.500069401E+00,-3.992141160E-08, 1.205272140E-11,-1.819015576E-15, 1.078576636E-19, -7.449939610E+02, 4.379180110E+00),(0.000)),dtype=object)
coefsH2O=np.array(((-3.947960830E+04, 5.755731020E+02, 9.317826530E-01, 7.222712860E-03,-7.342557370E-06,4.955043490E-09,-1.336933246E-12, -3.303974310E+04, 1.724205775E+01),(1.034972096E+06,-2.412698562E+03, 4.646110780E+00, 2.291998307E-03,-6.836830480E-07,9.426468930E-11,-4.822380530E-15, -1.384286509E+04,-7.978148510E+00),(-241826.000)),dtype=object)
coefsJetA=np.array(((-6.068695590E+05, 8.328259590E+03,-4.312321270E+01, 2.572390455E-01,-2.629316040E-04,1.644988940E-07,-4.645335140E-11, -7.606962760E+04, 2.794305937E+02),(1.858356102E+07,-7.677219890E+04, 1.419826133E+02,-7.437524530E-03, 5.856202550E-07,1.223955647E-11,-3.149201922E-15, 4.221989520E+05,-8.986061040E+02),(-249657.000)),dtype=object)#диапазон применимости для газообразного керосина от 273К!!!
coefsJetALiquid=np.array(((-4.218262130E+05,-5.576600450E+03, 1.522120958E+02,-8.610197550E-01, 3.071662234E-03,-4.702789540E-06, 2.743019833E-09, -3.238369150E+04,-6.781094910E+02),(-4.218262130E+05,-5.576600450E+03, 1.522120958E+02,-8.610197550E-01, 3.071662234E-03,-4.702789540E-06, 2.743019833E-09, -3.238369150E+04,-6.781094910E+02),(-303403.0E+01)),dtype=object) #диапазон температуры топлива от 220 до 550К. В данном случае также для жидкого керосина для унификации один и тот же массив коэффициентов повторяется два раза, но тем не менее, жидкий керосин можно считать только в указаном диапазоне от 220 до 550К!!! иначе результаты будут неадекватными
coefsCH4=np.array(((-1.766850998E+05,2.786181020E+03,-1.202577850E+01,3.917619290E-02,-3.619054430E-05,2.026853043E-08,-4.976705490E-12,-2.331314360E+04, 8.904322750E+01),(3.730042760E+06,-1.383501485E+04,2.049107091E+01,-1.961974759E-03,4.727313040E-07,-3.728814690E-11,1.623737207E-15,7.532066910E+04,-1.219124889E+02),(-74600.000)),dtype=object) 
coefsCH3=np.array(((-2.876188806E+04,5.093268660E+02,2.002143949E-01,1.363605829E-02,-1.433989346E-05,1.013556725E-08,-3.027331936E-12, 1.408271825E+04, 2.022772791E+01),(2.760802663E+06,-9.336531170E+03, 1.487729606E+01,-1.439429774E-03, 2.444477951E-07, -2.224555778E-11, 8.395065760E-16, 7.481809480E+04,-7.919682400E+01),(146658.040)),dtype=object)
coefsAir=np.array(((1.009950160E+04,-1.968275610E+02,5.009155110E+00,-5.761013730E-03,1.066859930E-05,-7.940297970E-09,2.185231910E-12,-1.767967310E+02,-3.921504225E+00),(2.415214430E+05,-1.257874600E+03, 5.144558670E+00,-2.138541790E-04, 7.065227840E-08, -1.071483490E-11, 6.577800150E-16, 6.462263190E+03,-8.147411905E+00),(-125.530)),dtype=object)

# coefs_v=np.array((coefsN2,coefsO2,coefsAr,coefsCO2,coefsH2O,coefsJetA,coefsJetALiquid))
coefs_dict={'N2':coefsN2,
           'O2':coefsO2,
           'CO2':coefsCO2,
           'Ar':coefsAr,
           'H2O':coefsH2O,
           'JetA_gas':coefsJetA,
           'JetA_liquid':coefsJetALiquid,
           'CH4':coefsCH4,
           'CH3':coefsCH3,
            'Air':coefsAir}

R_N2=Runiversal/MolW_N2#J/(kg-K)
R_O2=Runiversal/MolW_O2
R_CO2=Runiversal/MolW_CO2
R_Ar=Runiversal/MolW_Ar
R_H2O=Runiversal/MolW_H2O
R_JetA=Runiversal/MolW_JetA
R_CH4=Runiversal/MolW_CH4
R_CH3=Runiversal/MolW_CH3
R_Air=Runiversal/MolW_Air
R_dict={'N2':R_N2,
           'O2':R_O2,
           'CO2':R_CO2,
           'Ar':R_Ar,
           'H2O':R_H2O,
           'JetA_gas':R_JetA,
           'JetA_liquid':R_JetA,
           'CH4':R_CH4,
           'CH3':R_CH3,
            'Air':R_Air}
# R_v=np.array((R_N2,R_O2,R_Ar,R_CO2,R_H2O,R_JetA,R_JetA))

#функция, с помощью которой проверяется какие вещества будут использоваться в расчете, и которая подготавливает массивы исходных данных:
# used_gases=('N2','O2','Ar','CO2','H2O','JetA_gas','JetA_liquid')
# used_gases=('N2','O2','Ar','CO2','H2O','CH4')

def prepare_thermodynamics(used_gases):
    global coefs_v
    coefs_v=[]
    global MolW_v
    MolW_v=[]
    global R_v
    R_v=[]
    global T_critical_v
    T_critical_v=[]
    global Ro_critical_v
    Ro_critical_v=[]
    global Number_of_components #число компонентов в рассматриваемой смеси, это глобальная переменная нужна далее в алгоритмах узлов двигателя для корректного расчета своств рабочего тела. Ставим то число, которое считаем нужным. напрмер для двигателя на керосине это 7: 5 - составляющие воздуха, 1 - жидкий керосин, 1 - газообразный керосин TODO!!! нужно избавиться от этой переменной, нужно чтобы пользователь задавал только состав смеси газа
    Number_of_components=len(used_gases)
    if 'H2O' in used_gases:
        global index_of_H2O
        index_of_H2O=used_gases.index('H2O')
    for key in used_gases:
        coefs_v.append(coefs_dict[key])
        MolW_v.append(MolW_dict[key])
        R_v.append(R_dict[key])
        T_critical_v.append(T_critical_dict[key])
        Ro_critical_v.append(Ro_critical_dict[key])

#далее функции для вычисления свойств отдельных веществ
#у функции H и  S ниже "не выставлен" ноль, поэтому использовать их для вычислдения абсолютного значения нельзя! Диапазон применимости функций от 200 до 6000К
#TODO!!! как выяснилось тут: https://stackoverflow.com/questions/52603487/speed-comparison-numpy-vs-python-standard использовать numpy есть смысл только при работе с очень большими массивами, а для обычных операций со скалярами и маленькими массивами стандартные функции питон работают гораздо быстрее! Это надо проверить!
#нужно попробовать задавать массивы с полиномами и с составом через стандартные питоновские функции array.array и сравнить с numpy
#или как вариант вообще пеерписать весь файл thermodynamics.py на Си++, т.к. к функциям из этого файла идут тысячи обращений и как правило производительность упирается в него

def _Cp(T,coefs,R):
    rez=0
    if T<1000:
        v=coefs[0]
    elif T>=1000:
        v=coefs[1]
    for i,c in enumerate(v[:-2]):
        rez+=c*(T**(i-2))
    return rez*R #Дж/моль/К или Дж/кг/К, смотря какую R подставлять в формулу

def _H(T,coefs,R):
    rez=0
    if T<1000:
        v=coefs[0]
    elif T>=1000:
        v=coefs[1]
    rez=-v[0]*(T**(-2))+v[1]*np.log(T)/T+v[2]+v[3]*T/2+v[4]*(T**2)/3+v[5]*(T**3)/4+v[6]*(T**4)/5+v[7]/T
    return rez*R*T #Дж/моль или Дж/кг, смотря какую R подставлять в формулу

def _Sf(T,coefs,R):
    rez=0
    if T<1000:
        v=coefs[0]
    elif T>=1000:
        v=coefs[1]
    rez=-v[0]*(T**(-2))/2-v[1]/T+v[2]*np.log(T)+v[3]*T+v[4]*(T**2)/2+v[5]*(T**3)/3+v[6]*(T**4)/4+v[8]
    return rez*R #Дж/моль/К или Дж/кг/К, смотря какую R подставлять в формулу

#функция для преобразования вектора с мольными долями смеси в массовые доли
def mass_comp(mole_comp):
    EntireMolW=0
    for molP,molW in zip(mole_comp,MolW_v):
        EntireMolW+=molP*molW
    rez=np.array([molP*molW/EntireMolW for molP,molW in zip(mole_comp,MolW_v)])
    return rez
#функция для преобразования вектора с массовыми долями смеси в мольные доли
def mole_comp(mass_comp):
    moleP=np.array([massP/molW for massP,molW in zip(mass_comp,MolW_v)])
    moleEntire=np.sum(moleP)
    return moleP/moleEntire
#молярная масса смеси
def MolW_mix(mole_comp):
    v=[mole_comp_i*MolW_v_i for mole_comp_i,MolW_v_i in zip(mole_comp,MolW_v)]
    rez=np.sum(v)
    return rez
#газовая постоянная смеси
def R_mix(mass_comp):
    return Runiversal/MolW_mix(mole_comp(mass_comp))

#теплоемкость жидкого керосина JetA
#def Cp_JetALiquid(T): #допустимый диапазон от 220 до 550 К
#    rez=0
#    v=coefsJetALiquid[0]
#    for i,c in enumerate(v[:-2]):
#        rez+=c*(T**(i-2))
#    return rez*R_JetA #Дж/кг/К, 
##энтальпия жидкого керосина JetA
#def H_JetALiquid(T): #допустимый диапазон от 220 до 550 К
#    rez=0
#    v=coefsJetALiquid[0]
#    rez=-v[0]*(T**(-2))+v[1]*np.log(T)/T+v[2]+v[3]*T/2+v[4]*(T**2)/3+v[5]*(T**3)/4+v[6]*(T**4)/5+v[7]/T
#    return rez*R_JetA*T #Дж/кг

#вычисление теплоемкости смеси

def Cp(T,mass_comp):
    rez=0
    for mass_comp_i,coefs_i,R_i in zip(mass_comp,coefs_v,R_v):
        rez+=mass_comp_i*_Cp(T,coefs_i,R_i)
    return rez
#вычисление энтальпии смеси

def H(T,mass_comp):
    rez=0
    for mass_comp_i,coefs_i,R_i in zip(mass_comp,coefs_v,R_v):
        rez+=mass_comp_i*_H(T,coefs_i,R_i)
    return rez

#вычисление квази-энтропии смеси
def Sf(T,mass_comp):
    rez=0
    for mass_comp_i,coefs_i,R_i in zip(mass_comp,coefs_v,R_v):
        rez+=mass_comp_i*_Sf(T,coefs_i,R_i)
    return rez
#вычисление настоящей энтропии смеси

def S(P,T,mass_comp,R):
    return Sf(T,mass_comp) - Sf(Tzero,mass_comp) - R*np.log(P / Pzero)
#коэффициент адиабаты
def k(T,mass_comp,R):
    return 1/(1-R/Cp(T,mass_comp))

#вычисление температуры смеси по ее теплоемкости
def T_thru_Cp(Cp_value,mass_comp,x_min,x_max):
    func=lambda T: Cp(T,mass_comp)-Cp_value
    return optimize.brentq(func,x_min,x_max,disp=True)

#вычисление температуры смеси по ее энтальпии
def T_thru_H(H_value,mass_comp,x_min,x_max):
    func=lambda T: H(T,mass_comp)-H_value
    return optimize.brentq(func,x_min,x_max,disp=True)

#вычисление температуры смеси по ее энтропии
def T_thru_S(S_value,mass_comp,x_min,x_max):
    func=lambda T: Sf(T,mass_comp)-S_value
    return optimize.brentq(func,x_min,x_max,disp=True)

#в некоторых задачах необходимо вычислить полные параметры по заданным статическим и безразмерной скорости лямбда: TODO!!! это довольно редко используемая но тяжелая функция, тяжелая потому, что здесь в нескольких местах идет обращенеи к функции Critical_Ts - надо упроститиь!
def T_thru_HsTsVcorr(Hs,Ts,V_corr,k_val,mass_comp, R):
    func=lambda Tx: H(Tx,mass_comp)-(Hs+V_corr*V_corr*k(Critical_Ts(Tx,mass_comp,R), mass_comp, R)*R*Critical_Ts(Tx,mass_comp,R)/2)
    _T=Ts/Tau_classic(k_val, V_corr) #эта строка нужна для приближенной оценки верхнего интервала, вннутри котрого нужно искать полную температуру.
    return optimize.root_scalar(func, method='secant', x0=_T,x1=1.00001*_T).root

#вычисление полной энтальпии по известной статической и скорости
def H_thru_HsV(Hs,V):
    return Hs+V*V/2

#поиск конечного давления при известных начальных давлении и температуры и конечной температуре при изоэнтропическом процессе
def P2_thru_P1T1T2(P1,T1,T2,mass_comp,R): #эта штука считает давление независимо от того, сверхкритический или докритический перепад
    return P1*np.exp((Sf(T2,mass_comp) - Sf(T1,mass_comp)) / R)
#поиск конечной температуры при известных начальных давлении и температуры и конечном давлении при изоэнтропическом процессе
def T2_thru_P1T1P2(P1,T1,P2,mass_comp,R,x_min,x_max): #эта штука считает температуру независимо от того, сверхкритический или докритический перепад
    const_value=Sf(T1,mass_comp) + R*np.log(P2 / P1)
    func=lambda T2: Sf(T2,mass_comp) - const_value
    return optimize.brentq(func,x_min,x_max,disp=True)
#поиск критической статической температуры по известным полным параметрам
def Critical_Ts(T,mass_comp,R):
    H_value=H(T,mass_comp)
    func=lambda Ts: (k(Ts,mass_comp,R)*R*Ts) - (2 * (H_value - H(Ts,mass_comp)))
    return optimize.brentq(func,180,T,disp=True)
#статическая температура через энтальпию и число маха
def Ts_thru_HM(H_value,M,mass_comp,R,x_min,x_max):
    func=lambda Ts: H_value-H(Ts,mass_comp)-(M*M*k(Ts,mass_comp,R)*R*Ts/2)
    return  optimize.brentq(func,x_min,x_max,disp=True)
#статическая температура через энтальпию и скорость потока
def Ts_thru_HV(H_value,V,mass_comp,R,x_min,x_max):
    func=lambda Ts: H_value-H(Ts,mass_comp)-((V**2)/2)
    return  optimize.brentq(func,x_min,x_max,disp=True)
#статическая температура через расход, полные давление и температуру и площадь. Если перепад сверхкритический, то считает из условия запирания

def Ts_thru_GPTHF(G,P,T,Hval,F,Ts_cr,mass_comp,R): #медленная функция, если есть возможность, лучше ее не использовать
    func=lambda Ts: G**2-(P2_thru_P1T1T2(P,T,Ts,mass_comp,R)/R/Ts*F)**2*(2*(Hval-H(Ts,mass_comp)))
    # Ts= optimize.brentq(func,Ts_cr,T,disp=True)
    try:
        Ts= optimize.toms748(func,Ts_cr,T,disp=True) #эта штука должна считать быстрее чем brentq
    except ValueError:
        try:
            Ts=optimize.newton(func,Ts_cr,disp=True)
        except ValueError:
            solverLog.error(f'Error! Ts_thru_GPTHF - cant find Ts. Possibly reason in supercritical drop: G={G},P={P},T={T},Hval={Hval},F={F},Ts_cr={Ts_cr}')
    
    return Ts

#давление насыщенных паров (Buck equation из википедии)
def P_sat_vapour1(T):
    T=T-273.15
    if T>=0:
        return 0.61121*np.exp((18.678-T/234.5)*(T/(257.14+T)))*1000
    else:
        return 0.61115*np.exp((23.036-T/333.7)*(T/(279.82+T)))*1000
#давление насыщенных паров (из Gas Turbine Performance)
def P_sat_vapour2(P,T):
    return((1.0007+(3.46e-5)*(P/1000))*0.61121*np.exp(17.502*(T-273.15)/(T-32.25))*1000)
#формула для расчета удельной влажности/влагосодержания/отношения массы пара к массе сухогов воздуха
def WAR(rel_hum, P, T, mass_comp_dry_air ): #за основуную формулу для поиска давления насыщенных паров примем пока что P_sat_vapour1, т.е. без учета давления
    if rel_hum==0:   #!!!важно знать, что эта формула работает правильно при температуре возуха менее 100 Цельсий!!! Потребностей в бОльших температурах не должно возникать! Иначе результат функциий может быть отрицательным, т.е. не физичным
        rez=0
    else:
        
        Pvap = P_sat_vapour1(T)*rel_hum
        if (P-Pvap)<0:
            text="В функции для вычисления влагосодержания температура при которой вычисляестя давление пара неадекватно большое. Tатм={T}, Pатм={P}, Pvapor={Pvap}, Hum={rel_hum}  Возможна ошибка."
            raise ValueError(text)
        Ro_vap=Pvap/T/R_H2O
        Ro_dry_air=(P-Pvap)/T/R_mix(mass_comp_dry_air)
        rez=((Ro_vap)/(Ro_dry_air)) #отношение плотностей пара и сухого! воздуха
    return rez

#расчет относительной влажности по влагосодержанию        
def Rel_humidity(WAR,P,T):
    Psat_vap = P_sat_vapour1(T) #давление насыщенного пара
    Pvap=P*WAR/(1+WAR)
    return Pvap/Psat_vap


def WAR_to_moist_air(WAR,mass_comp_dry_air):
    mass_comp_water=np.empty(Number_of_components) #!!!размер массива должен соответствовать числу составляющих газ
    mass_comp_water[:]=0
    mass_comp_water[index_of_H2O]=1 #!здесь важно чтобы индекс соответствоваол порядковому номеру воды в массиве mass_comp
    rez=mass_comp_dry_air*(1-WAR)+mass_comp_water*WAR
    return rez

def T_ISA(H):
    if H<11000:
        T=288.15-0.0065*H
    elif H<24994 and H>=11000:
        T=216.65
    elif H<30000 and H>=24994:
        T=216.65+0.0029892*(H-24994)
    return T

def P_ISA(H):
    if H<11000:
        P=101325*(288.15/T_ISA(H))**(-5.25588)
    elif H<24994 and H>=11000:
        P=22.63253/np.exp(0.000157689*(H-10998.1))
    elif H<30000 and H>=24994:
        P=2.5237*(216.65/T_ISA(H))**11.8
    return P

def Dyn_visc_klimov(T):
    _T=T/1000 #динамический коэффициент вязкости, определяемый для расчета поправок к характеристикам турбины по данным "Математическая модель двигателя 26.616.0004-2016ММ1", стр27
    return ((_T**1.274511)*np.exp(1.455223-0.3054685*_T))+1 #для расчета других двигателей лучше стоит убедиться в достоверности этой формулы или использовать данные из NASA

def Dyn_visc_sultanian(Ts): #Sultanian, app.B, p.322. result units = Pa*sec только для воздуха!
    return(4.112985e-6+5.052295e-8*Ts-1.43462e-11*Ts**2+2.591403e-15*Ts**3)

def Dyn_viscosity(Ts,mass_comp):#по методике interacting sphere model (как в CFX-solver) (Applications of Kinetic Gas Theories and Multiparameter Correlation for Prediction of Dilute Gas Viscosity and Thermal Conductivity)
    #TODO!!! эта функция очень! тяжелая, нужно сделать в prepare_thermodynamics штуку, которая будет аппроксимировать эту функцию в полином, к которому потом можно будет обращаться из программы
    #результат в системе СИ Па*сек
    Tc=0
    Vc=0
    Roc=0
    MolW=MolW_mix(mole_comp(mass_comp)) 
    for Roc_i,Tc_i,mass_comp_i in zip(Ro_critical_v,T_critical_v,mass_comp):
        # print(Roc_i,Tc_i,MolW_i,mass_comp_i)
        Tc+=Tc_i*mass_comp_i
        Roc+=Roc_i*mass_comp_i
    Vc=1000000*MolW/Roc #критический молярный объем в см3/моль             1000000*kg/mole*m3/kg = cm3/mole
    # print(Tc,Vc,MolW)
    T_=1.2593*Ts/Tc
    Omega=1.16145*T_**(-0.14874)+0.52487*np.exp(-0.7732*T_)+2.16178*np.exp(-2.43787*T_) #collision function
    Sigma=0.809*Vc**(1/3) #collision diameter в ангстремах
    return 2.669*np.sqrt(MolW*1000*Ts)/(Omega*Sigma*Sigma)*1e-6 #в формуле молярная масса д.б. в г/моль, поэтмоу *1000; 10e-6 - потому что в исходной формуле единицы измерения в микроПаскалях 
    


#TODO!!! сделать отдельный пул формул из классической газодинамики, подумать какие формулы нужны, в частности обязательно газо-динамические функции

def Ps_Pt(V_corr,k): #газодинамическая функция по классической формуле: отношение статического к полному давлению в зависимости от приведенной скорости
    a=k-1
    b=k+1
    rez=(1-a*V_corr**2/b)**(k/a)
    return rez

#функция для вычисления состава продуктов сгорания топлива по известному относительному расходу топлива
def RelativeFuelFlow2GasMixture(dry_air,stoichiometric_gas,rel_fuel_flow,L):
    Gfuel=rel_fuel_flow #отн расход топлива = расход топлива/расход воздуха
    Gair_burnt=Gfuel*L
    Gair_unburnt=1-Gair_burnt
    gas_mixture=(Gair_burnt+Gfuel)/(1+Gfuel)*stoichiometric_gas+Gair_unburnt/(1+Gfuel)*dry_air
    return gas_mixture

#классические газодинамические функции, которые в общем-то не совсем точны
def Pi_classic(k,V_corr): #отношение статического давления к полному
    return (1-V_corr*V_corr*((k-1)/(k+1)))**(k/(k-1))

def Tau_classic(k,V_corr): #отношенеи статической температуры  к полной
    return (1-V_corr*V_corr*((k-1)/(k+1)))

#теплопроводность воздуха по данным из гидры #TODO! возможно стоит ввести нормальный расчет теплопроводности для различных газов из нормальных источников
#температура в кельвинах, диапазон применимости хз?
def thermal_conductivity_air_hydra(T):
    return 0.0251*(T/273.15)**0.7754

#число Прандтля по данным Саши Себелева (сообщенеи в телеге 16.04.2021), температура в Кельвинах, диапазон применимости 300-1500К #TODO! возможно стоит ввести нормальный расчет Прандтля для различных газов из нормальных источников
def Prandtl_hydra(T):
    return 0.000016 * (T / 100) ** 3 - 0.00068 * (T / 100) ** 3 + 0.01 * (T / 100) ** 2 - 0.0554 * (T / 100) + 0.8

#класс для описания всех параметров изоэнтропического процесса без привязки к геометрии, т.е. условие сверхкритического/докритического потока здесь не производится
class IsentropicFlow():
    #!!!НУЖНО ПОРАБОТАТЬ НАД ОПТИМИЗАЦИЕЙ ЭТОГО КЛАССА, ТК ОН ОДИН ИЗ БАЗОВЫХ И К НЕМУ ИДЕТ СОТНИ И ТЫСЯЧИ ОБРАЩЕНИЙ
    list_of_parameters=['name','mass_comp','R','P','T','Ro','Cp','Cv','k','H','Sf','S','Ps','Ts','Ros','Cps','Cvs','ks','Hs',
                        'Sfs','Ss','Vsnd','flowdensity','pi','tau','q','Ts_cr','Ps_cr','Ros_cr','V_cr','k_cr','flowdensity_cr'] #здесь будут храниться множество параметров этого класса, которые нужно выводить в консоль по методу status
    def __init__(self,**parameters):
        #СНАЧАЛА ПЕРЕЧИСОЯЕМ ВСЕ ПАРАМЕТРЫ ИСПОЛЬЗУЕМЫЕ ДЛЯ ОПИСАНИЕ СОСТОЯНИЯ В СЕЧЕНИИ ПОТОКА
        #TODO!! попробовать использовать вместо скалярных значений массив ndarray c одним значением, это нужно для того, чтобы была возможность передавать значения по ссылке. По тестам если использовать массив с одним числом вместо скаляра на быстродействии это скажется только по части записи, чтение по скорости не отдичается. Зато передача параметра по ссылке может придать ускорение в други
        # self.was_edited = False #флаг для отслеживания того, была ли структура отредактирована в процессе выполнения метода
        self.calc_now=set() #по умолчанию множество пустое. Здесь хранятся имена переменных, которыек вычисляются в настоящий момент. Этот механизм нужен, чтобы недопуститить возникновения самозацикленных ссылок, т.к. например мы пытаемся вычислить P(T), далее алгоритм заходит в геттер T и пытается посчитаеть его значение через P, из-за этого заходит снова в геттер P - возникает вечный цикл
        self.init_data=set() #в множестве хранятся имена тех параметров, которые являются исходными данными, а не результатами расчета. Это сделано для того, чтобы была возможность сбрасывать результаты расчета
        self.init_data_parameters=parameters #словарь параметров переданных через инициализатор
        self._name = self.get_init_data('name')
        # 1 группа параметров зависящих только от состава смеси
        # self._mass_comp=np.empty(Number_of_components) #!!!размер массива должен соответствовать числу составляющих газ
        self._mass_comp=self.get_init_data('mass_comp')
        # if self.exist(self._mass_comp):
        #     self.init_data.add('mass_comp')
        self._R=self.get_init_data('R')
        # 2 группа полные параметры смеси газа
        self._P=self.get_init_data('P')
        self._T=self.get_init_data('T')
        self._Ro=self.get_init_data('Ro')
        self._Cp=self.get_init_data('Cp')
        self._Cv=self.get_init_data('Cv')
        self._k=self.get_init_data('k')
        self._H=self.get_init_data('H')
        self._Sf=self.get_init_data('Sf')
        self._S = self.get_init_data('S')
        # 3 группа статические параметры смеси газа
        self._Ps=self.get_init_data('Ps')
        self._Ts=self.get_init_data('Ts')
        self._Ros=self.get_init_data('Ros')
        self._Cps = self.get_init_data('Cps')
        self._Cvs = self.get_init_data('Cvs')
        self._ks = self.get_init_data('ks')
        self._Hs=self.get_init_data('Hs')
        self._Sfs=self.get_init_data('Sfs')
        self._Ss= self.get_init_data('Ss')
        self._Vsnd=self.get_init_data('Vsnd')
        self._flowdensity=self.get_init_data('flowdensity')
        # 4 газодинамические параметры смеси газа
        self._pi = self.get_init_data('pi') 
        self._tau = self.get_init_data('tau')
        self._q = self.get_init_data('q')
        # 5 группа критические параметры
        self._Ts_cr=self.get_init_data('Ts_cr')
        self._Ps_cr=self.get_init_data('Ps_cr')
        self._Ros_cr=self.get_init_data('Ros_cr')
        self._V_cr=self.get_init_data('V_cr')
        self._k_cr=self.get_init_data('k_cr')
        self._flowdensity_cr=self.get_init_data('flowdensity_cr')
    
    #вспомогательные функции:    
    def exist(self,*val):
        # return False if np.isnan(np.sum(val)) else True
        return False if np.any(np.isnan(val)) else True
    
    def not_exist(self,*val):
        # return True if np.isnan(np.sum(val)) else False
        return True if np.all(np.isnan(val)) else False
    
    def get_init_data(self,par): #функция для получения величин исходных параметров через инициализатор экземпляра класса
        
        if par=='name':
            _x = self.init_data_parameters.get(par,'')
            self.init_data.add(par)
        elif par=='mass_comp':
            _x = self.init_data_parameters.get(par,np.full(Number_of_components,np.nan))
            if self.exist(_x):
                self.init_data.add(par)
        else:
            _x = self.init_data_parameters.get(par,np.nan)
            if self.exist(_x):
                self.init_data.add(par)
        return _x
        
    def reset_results(self):
        for key in self.__dict__.keys():
            _key=key[1:] 
            if key[0] == '_' and _key not in self.init_data: #TODO!!! придумать механизм, который бы заменил список self.init_data на что-то типа bit set, т.к. битсеты работают быстрее, что для данной функции критично, т.к. она вызывается очень часто
                if _key=='mass_comp':
                    self._mass_comp.fill(np.nan)
                else:
                    setattr(self,key,np.nan)
                
    #показать ошибку в том случае, если первый элемент из спика *parameters является np.nan
    def show_error(self,unknown_parameter,text_to_show=''):
        if self.not_exist(unknown_parameter):
           solverLog.debug(text_to_show+' '+str(unknown_parameter))
           # raise SystemExit
           
    def check_calc_now(self,*val): #проверяем значения из списка val в множестве self.calc_val, если значение есть, то возвращаем False
        for e in val:
            if e in self.calc_now:
                return False
        return True
    
    #ОПРЕДЕЛИМ МЕТОДЫ КЛАССА IsentropicProcess
    @property
    def name(self):
        return self._name   
    @name.setter
    def name(self,x):
        self._name=x
    
    @property
    def R(self):
        if self.not_exist(self._R) and self.exist(self.mass_comp):
            # self.calc_now.add('R')
            self._R = R_mix(self.mass_comp)
            # self.was_edited = True
            # self.calc_now.remove('R')
        self.show_error(self._R,text_to_show='ERROR! Isentropic Flow obj: {parent}: noncalculable variable: R ='.format(parent=self._name))
        return self._R   
    @R.setter
    def R(self,x):
        self._R=x
        if np.isnan(x):
            self.init_data.discard('R')
        else:
            self.init_data.add('R')
        
    @property
    def mass_comp(self):
        self.show_error(self._mass_comp,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: mass_comp ='.format(parent=self._name))
        return self._mass_comp
    @mass_comp.setter
    def mass_comp(self,x):
        if isinstance(x,float):
            self._mass_comp[:] = x
            if np.isnan(x):
                self.init_data.discard('mass_comp')
            else:
                self.init_data.add('mass_comp')
        elif isinstance(x,np.ndarray):
            self._mass_comp = x.copy()
            if all(np.isnan(x)):
                self.init_data.discard('mass_comp')
            else:
                self.init_data.add('mass_comp')
        
    @property
    def P(self):
        if self.not_exist(self._P):
            self.calc_now.add('P')
            TRo=self.check_calc_now('T','Ro') and self.exist(self.R,self.T,self.Ro)
            TTsPs=self.check_calc_now('T','Ts','Ps') and self.exist(self.T,self.Ts,self.Ps) and self.exist(self.mass_comp)
            if TRo and TTsPs:
                solverLog.warning('WARNING! ' + self._name + ' Overdetermined source data for calculating of P.')
                raise SystemExit
            if TRo:
                self._P = self.R*self.T*self.Ro
                # self.was_edited = True
            if TTsPs:
                if self.Ts>=self.T:
                    solverLog.error('ERROR! ' + self._name + ': calculating of P: Ts({Ts})>T({T})'.format(T=self.T,Ts=self.Ts))
                self._P=P2_thru_P1T1T2(self.Ps,self.Ts,self.T,self.mass_comp,self.R)
                # self.was_edited = True
            self.calc_now.remove('P')
        self.show_error(self._P,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: P ='.format(parent=self._name))
        return self._P
    @P.setter
    def P(self,x):
        self._P=x
        if np.isnan(x):
            self.init_data.discard('P')
        else:
            self.init_data.add('P')

    @property
    def T(self):
        if self.not_exist(self._T):
            self.calc_now.add('T')
            PRo=self.check_calc_now('P','Ro') and self.exist(self.R,self.P,self.Ro)
            H=self.check_calc_now('H') and self.exist(self.H) and self.exist(self.mass_comp)
            PTsPs=self.check_calc_now('P','Ps','Ts') and self.exist(self.P,self.Ts,self.Ps) and self.exist(self.mass_comp)
            if int(PRo) + int(H) + int(PTsPs)>1:
                solverLog.warning('WARNING! ' + self._name + ' Overdetermined source data for calculating of T.')
                raise SystemExit
            if PRo: #ищем температуру из P/Ro=R*T
                self._T = self.P / self.Ro / self.R
                # self.was_edited = True
            elif H:
                self._T=T_thru_H(self.H,self.mass_comp,150,3000)
                # self.was_edited = True
            elif PTsPs:
                if self.Ps>=self.P:
                    solverLog.error('ERROR! ' + self._name + ': calculating of T: Ps({Ps})>P({P})'.format(P=self.P,Ps=self.Ps))
                self._T=T2_thru_P1T1P2(self.Ps,self.Ts,self.P,self.mass_comp,self.R,150,3000)
                # self.was_edited = True
            self.calc_now.remove('T')
        self.show_error(self._T,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: T ='.format(parent=self._name))
        return self._T
    @T.setter
    def T(self,x):
        self._T=x
        if np.isnan(x):
            self.init_data.discard('T')
        else:
            self.init_data.add('T')

    @property
    def Ro(self):
        if self.not_exist(self._Ro):
            self.calc_now.add('Ro')
            if self.check_calc_now('P','T') and self.exist(self.P, self.R, self.T):
                self._Ro = self.P / self.R / self.T
                # self.was_edited = True
            self.calc_now.remove('Ro')
        self.show_error(self._Ro,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Ro ='.format(parent=self._name))
        return self._Ro
    @Ro.setter
    def Ro(self,x):
        self._Ro=x
        if np.isnan(x):
            self.init_data.discard('Ro')
        else:
            self.init_data.add('Ro')
        
    @property
    def Cp(self):
        if self.not_exist(self._Cp) and self.exist(self.T) and self.exist(self.mass_comp):
            self._Cp = Cp(self.T, self.mass_comp)
            # self.was_edited = True
        self.show_error(self._Cp,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Cp ='.format(parent=self._name))
        return self._Cp
    @Cp.setter
    def Cp(self,x):
        self._Cp=x
        if np.isnan(x):
            self.init_data.discard('Cp')
        else:
            self.init_data.add('Cp')                       

    @property
    def Cv(self):
        if self.not_exist(self._Cv) and self.exist(self.Cp, self.k):
            self._Cv = self.Cp / self.k
            # self.was_edited = True
        self.show_error(self._Cv,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Cv ='.format(parent=self._name))
        return self._Cv
    @Cv.setter
    def Cv(self,x):
        self._Cv=x
        if np.isnan(x):
            self.init_data.discard('Cv')
        else:
            self.init_data.add('Cv')
    
    @property
    def k(self):
        if self.not_exist(self._k) and self.exist(self.T, self.R) and self.exist(self.mass_comp):
            self._k = k(self.T, self.mass_comp, self.R)
            # self.was_edited = True
        self.show_error(self._k,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: k ='.format(parent=self._name))
        return self._k
    @k.setter
    def k(self,x):
        self._k=x
        if np.isnan(x):
            self.init_data.discard('k')
        else:
            self.init_data.add('k')

    @property
    def H(self):
        if self.not_exist(self._H):
            self.calc_now.add('H')
            if self.check_calc_now('T') and self.exist(self.T) and self.exist(self.mass_comp):
                self._H = H(self.T, self.mass_comp)
                # self.was_edited = True
            self.calc_now.remove('H')
        self.show_error(self._H,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: H ='.format(parent=self._name))
        return self._H
    @H.setter
    def H(self,x):
        self._H=x
        if np.isnan(x):
            self.init_data.discard('H')
        else:
            self.init_data.add('H')

    @property
    def Sf(self):
        if self.not_exist(self._Sf):
            self.calc_now.add('Sf')
            if self.check_calc_now('T') and self.exist(self.T) and self.exist(self.mass_comp):
                self._Sf = Sf(self.T, self.mass_comp)
                # self.was_edited = True
            self.calc_now.remove('Sf')
        self.show_error(self._Sf,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Sf ='.format(parent=self._name))
        return self._Sf
    @Sf.setter
    def Sf(self,x):
        self._Sf=x
        if np.isnan(x):
            self.init_data.discard('Sf')
        else:
            self.init_data.add('Sf')

    @property
    def S(self):
        if self.not_exist(self._S):
            self.calc_now.add('S')
            if self.check_calc_now('P','T') and self.exist(self.P,self.T,self.R) and self.exist(self.mass_comp):
                self._S = S(self.P,self.T,self.mass_comp,self.R)
                # self.was_edited = True
            self.calc_now.remove('S')
        self.show_error(self._S,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: S ='.format(parent=self._name))
        return self._S
    @S.setter
    def S(self,x):
        self._S=x
        if np.isnan(x):
            self.init_data.discard('S')
        else:
            self.init_data.add('S')
        
    @property
    def Ps(self):
        if self.not_exist(self._Ps):
            self.calc_now.add('Ps')
            TsPT= self.check_calc_now('P','T','Ts') and self.exist(self.P, self.T,self.R, self.Ts) and self.exist(self.mass_comp)
            TsRos=self.check_calc_now('Ros','Ts') and self.exist(self.Ros, self.R, self.Ts)
            if int(TsPT) + int(TsRos)>1:
                solverLog.warning('WARNING! ' + self._name + ' Overdetermined source data for calculating of Ps.')
                raise SystemExit
            if TsPT: #ищем давление через энтропию, дельта энтропии = 0
                if self.Ts>=self.T:
                    solverLog.error('ERROR! ' + self._name + ': calculating Ps: Ts({Ts})>T({T})'.format(T=self.T,Ts=self.Ts)) 
                self._Ps = P2_thru_P1T1T2(self.P, self.T, self.Ts, self.mass_comp,self.R)
                # self.was_edited = True
            elif TsRos: #ищем давление из P/Ro=R*T
                self._Ps = self.R*self.Ts*self.Ros
                # self.was_edited = True
            self.calc_now.remove('Ps')
        self.show_error(self._Ps,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Ps ='.format(parent=self._name))
        return self._Ps
    @Ps.setter
    def Ps(self,x):
        self._Ps=x
        if np.isnan(x):
            self.init_data.discard('Ps')
        else:
            self.init_data.add('Ps')
            
    @property
    def Ts(self):
        if self.not_exist(self._Ts):
            self.calc_now.add('Ts')
            RosPs=self.check_calc_now('Ros','Ps') and self.exist(self.Ros, self.Ps, self.R)
            TPPs=self.check_calc_now('T','P','Ps') and self.exist(self.T, self.P, self.R, self.Ps) and self.exist(self.mass_comp)
            Hs=self.check_calc_now('Hs') and self.exist(self.Hs) and self.exist(self.mass_comp)
            if int(RosPs) + int(TPPs) + int(Hs)>1:
                solverLog.warning('WARNING! ' + self._name + ' Calculating Ts: Overdetermined source data')
            if RosPs:
                self._Ts = self.Ps / self.Ros / self.R
                # self.was_edited = True
            elif TPPs:
                if self.Ps>=self.P:
                    solverLog.error('ERROR! ' + self._name + ': calculate_Ts: Ps({Ps})>P({P})'.format(P=str(self.P),Ps=str(self.Ps)))         
                self._Ts=T2_thru_P1T1P2(self.P,self.T,self.Ps,self.mass_comp,self.R,150,self.T)
                # self.was_edited = True
            elif Hs:
                self._Ts = T_thru_H(self.Hs, self.mass_comp,150,self.T)
                # self.was_edited = True      
            self.calc_now.remove('Ts')
        self.show_error(self._Ts,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Ts ='.format(parent=self._name))
        return self._Ts
    @Ts.setter
    def Ts(self,x):
        self._Ts=x
        if np.isnan(x):
            self.init_data.discard('Ts')
        else:
            self.init_data.add('Ts')
        
    @property
    def Ros(self):
        if self.not_exist(self._Ros):
            self.calc_now.add('Ros')
            if self.check_calc_now('Ps','Ts') and self.exist(self.Ps, self.R, self.Ts):
                self._Ros = self.Ps / self.R / self.Ts
                # self.was_edited = True
            self.calc_now.remove('Ros')
        self.show_error(self._Ros,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Ros ='.format(parent=self._name))
        return self._Ros
    @Ros.setter
    def Ros(self,x):
        self._Ros=x
        if np.isnan(x):
            self.init_data.discard('Ros')
        else:
            self.init_data.add('Ros')
        
    @property
    def Cps(self):
        if self.not_exist(self._Cps) and self.exist(self.Ts) and self.exist(self.mass_comp):
            self._Cps = Cp(self.Ts, self.mass_comp)
            # self.was_edited = True
        self.show_error(self._Cps,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Cps ='.format(parent=self._name))
        return self._Cps
    @Cps.setter
    def Cps(self,x):
        self._Cps=x
        if np.isnan(x):
            self.init_data.discard('Cps')
        else:
            self.init_data.add('Cps')

    @property
    def Cvs(self):
        if self.not_exist(self._Cvs) and self.exist(self.Cps, self.k):
            self._Cvs = self.Cps / self.k
            # self.was_edited = True
        self.show_error(self._Cvs,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Cvs ='.format(parent=self._name))
        return self._Cvs
    @Cvs.setter
    def Cvs(self,x):
        self._Cvs=x
        if np.isnan(x):
            self.init_data.discard('Cvs')
        else:
            self.init_data.add('Cvs')           

    @property
    def ks(self):
        if self.not_exist(self._ks) and self.exist(self.Ts, self.R) and self.exist(self.mass_comp):
            self._ks = k(self.Ts, self.mass_comp, self.R)
            # self.was_edited = True
        self.show_error(self._ks,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: ks ='.format(parent=self._name))
        return self._ks
    @ks.setter
    def ks(self,x):
        self._ks=x
        if np.isnan(x):
            self.init_data.discard('ks')
        else:
            self.init_data.add('ks')
        
    @property
    def Hs(self):
        if self.not_exist(self._Hs):
            self.calc_now.add('Hs')
            if self.check_calc_now('Ts') and self.exist(self.Ts) and self.exist(self.mass_comp):
                self._Hs = H(self.Ts, self.mass_comp)
                # self.was_edited = True
            self.calc_now.remove('Hs')
        self.show_error(self._Hs,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Hs ='.format(parent=self._name))
        return self._Hs
    @Hs.setter
    def Hs(self,x):
        self._Hs=x
        if np.isnan(x):
            self.init_data.discard('Hs')
        else:
            self.init_data.add('Hs')
        
    @property
    def Sfs(self):
        if self.not_exist(self._Sfs):
            self.calc_now.add('Sfs')
            if self.check_calc_now('Ts') and self.exist(self.Ts) and self.exist(self.mass_comp):
                self._Sfs = Sf(self.Ts, self.mass_comp)
                # self.was_edited = True
            self.calc_now.remove('Sfs')
        self.show_error(self._Sfs,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Sfs ='.format(parent=self._name))
        return self._Sfs
    @Sfs.setter
    def Sfs(self,x):
        self._Sfs=x
        if np.isnan(x):
            self.init_data.discard('Sfs')
        else:
            self.init_data.add('Sfs')

    @property
    def Ss(self):
        if self.not_exist(self._Ss):
            self.calc_now.add('Ss')
            if self.check_calc_now('Ps','Ts') and self.exist(self.Ps,self.Ts,self.R) and self.exist(self.mass_comp):
                self._Ss = S(self.Ps,self.Ts,self.mass_comp,self.R)
                # self.was_edited = True
            self.calc_now.remove('Ss')
        self.show_error(self._Ss,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Ss ='.format(parent=self._name))
        return self._Ss
    @Ss.setter
    def Ss(self,x):
        self._Ss=x
        if np.isnan(x):
            self.init_data.discard('Ss')
        else:
            self.init_data.add('Ss')
        
    @property
    def Vsnd(self):
        if self.not_exist(self._Vsnd):
            self.calc_now.add('Vsnd')
            if self.check_calc_now('Ts') and self.exist(self.ks, self.R, self.Ts):
                self._Vsnd=np.sqrt(self.ks*self.R*self.Ts)
                # self.was_edited = True
            self.calc_now.remove('Vsnd')
        self.show_error(self._Vsnd,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Vsnd ='.format(parent=self._name))
        return self._Vsnd
    @Vsnd.setter
    def Vsnd(self,x):
        self._Vsnd=x 
        if np.isnan(x):
            self.init_data.discard('Vsnd')
        else:
            self.init_data.add('Vsnd')
        
    @property
    def flowdensity(self):
        if self.not_exist(self._flowdensity):
            self.calc_now.add('flowdensity')
            if self.check_calc_now('Ros','H','Hs') and self.exist(self.Ros,self.H,self.Hs):
                if self.Hs>self.H:
                    solverLog.error('ERROR! ' + self._name + ': calculate_flowdensity: Hs({Hs})>H({H})'.format(H=self.H,Hs=self.Hs))       
                self._flowdensity=self.Ros*np.sqrt(2*(self.H-self.Hs))
                # self.was_edited = True
            self.calc_now.remove('flowdensity')
        self.show_error(self._flowdensity,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: flowdensity ='.format(parent=self._name))
        return self._flowdensity
    @flowdensity.setter
    def flowdensity(self,x):
        self._flowdensity=x
        if np.isnan(x):
            self.init_data.discard('flowdensity')
        else:
            self.init_data.add('flowdensity')
        
    @property
    def pi(self):
        if self.not_exist(self._pi) and self.exist(self.Ps, self.P):
            self._pi = self.Ps / self.P
            # self.was_edited = True
        self.show_error(self._pi,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: pi ='.format(parent=self._name))
        return self._pi
    @pi.setter
    def pi(self,x):
        self._pi=x
        if np.isnan(x):
            self.init_data.discard('pi')
        else:
            self.init_data.add('pi')
        
    @property
    def tau(self):
        if self.not_exist(self._tau) and self.exist(self.Ts, self.T):
            self._tau = self.Ts / self.T
            # self.was_edited = True
        self.show_error(self._tau,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: tau ='.format(parent=self._name))
        return self._tau
    @tau.setter
    def tau(self,x):
        self._tau=x
        if np.isnan(x):
            self.init_data.discard('tau')
        else:
            self.init_data.add('tau')
        
    @property
    def q(self):
        if self.not_exist(self._q) and self.exist(self.flowdensity, self.flowdensity_cr):
            self._q = self.flowdensity/ self.flowdensity_cr
            # self.was_edited = True
        self.show_error(self._q,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: q ='.format(parent=self._name))
        return self._q
    @q.setter
    def q(self,x):
        self._q=x
        if np.isnan(x):
            self.init_data.discard('q')
        else:
            self.init_data.add('q')

    @property
    def Ts_cr(self):
        if self.not_exist(self._Ts_cr):
            self.calc_now.add('Ts_cr')
            if self.check_calc_now('T') and self.exist(self.T,self.R) and self.exist(self.mass_comp):
                self._Ts_cr=Critical_Ts(self.T,self.mass_comp,self.R)
                # self.was_edited = True
            self.calc_now.remove('Ts_cr')
        self.show_error(self._Ts_cr,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Ts_cr ='.format(parent=self._name))
        return self._Ts_cr
    @Ts_cr.setter
    def Ts_cr(self,x):
        self._Ts_cr=x
        if np.isnan(x):
            self.init_data.discard('Ts_cr')
        else:
            self.init_data.add('Ts_cr')
        
    @property
    def Ps_cr(self):
        if self.not_exist(self._Ps_cr):
            self.calc_now.add('Ps_cr')
            if self.check_calc_now('P','T','Ts_cr') and self.exist(self.P,self.T,self.Ts_cr,self.R) and self.exist(self.mass_comp):
                self._Ps_cr=P2_thru_P1T1T2(self.P,self.T,self.Ts_cr,self.mass_comp,self.R)
                # self.was_edited = True
            self.calc_now.remove('Ps_cr')
        self.show_error(self._Ps_cr,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Ps_cr ='.format(parent=self._name))
        return self._Ps_cr
    @Ps_cr.setter
    def Ps_cr(self,x):
        self._Ps_cr=x
        if np.isnan(x):
            self.init_data.discard('Ps_cr')
        else:
            self.init_data.add('Ps_cr')
        
    @property
    def Ros_cr(self):
        if self.not_exist(self._Ros_cr):
            self.calc_now.add('Ros_cr')
            if self.check_calc_now('Ps_cr','Ts_cr') and self.exist(self.Ps_cr,self.Ts_cr,self.R):
                self._Ros_cr=self.Ps_cr/self.Ts_cr/self.R
                # self.was_edited = True
            self.calc_now.remove('Ros_cr')
        self.show_error(self._Ros_cr,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Ros_cr ='.format(parent=self._name))
        return self._Ros_cr
    @Ros_cr.setter
    def Ros_cr(self,x):
        self._Ros_cr=x
        if np.isnan(x):
            self.init_data.discard('Ros_cr')
        else:
            self.init_data.add('Ros_cr')

    @property
    def k_cr(self):
        if self.not_exist(self._k_cr) and self.exist(self.Ts_cr, self.R) and self.exist(self.mass_comp):
            self._k_cr = k(self.Ts_cr, self.mass_comp, self.R)
            # self.was_edited = True
        self.show_error(self._k_cr,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: k_cr ='.format(parent=self._name))
        return self._k_cr
    @k_cr.setter
    def k_cr(self,x):
        self._k_cr=x
        if np.isnan(x):
            self.init_data.discard('k_cr')
        else:
            self.init_data.add('k_cr')
        
    @property
    def V_cr(self):
        if self.not_exist(self._V_cr):
            self.calc_now.add('V_cr')
            if self.check_calc_now('Ts_cr') and self.exist(self.k_cr,self.R,self.Ts_cr):
                self._V_cr = np.sqrt(self.k_cr*self.R*self.Ts_cr)
                # self.was_edited = True
            self.calc_now.remove('V_cr')
        self.show_error(self._V_cr,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: V_cr ='.format(parent=self._name))
        return self._V_cr
    @V_cr.setter
    def V_cr(self,x):
        self._V_cr=x
        if np.isnan(x):
            self.init_data.discard('V_cr')
        else:
            self.init_data.add('V_cr')
        
    @property
    def flowdensity_cr(self):
        if self.not_exist(self._flowdensity_cr):
            self.calc_now.add('flowdensity_cr')
            if self.check_calc_now('V_cr','Ros_cr') and self.exist(self.V_cr,self.Ros_cr):
                self._flowdensity_cr = self.V_cr*self.Ros_cr
                # self.was_edited = True
            self.calc_now.remove('flowdensity_cr')
        self.show_error(self._flowdensity_cr,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: flowdensity_cr ='.format(parent=self._name))
        return self._flowdensity_cr
    @flowdensity_cr.setter
    def flowdensity_cr(self,x):
        self._flowdensity_cr=x
        if np.isnan(x):
            self.init_data.discard('flowdensity_cr')
        else:
            self.init_data.add('flowdensity_cr')

            
    def status(self): #вывод в консоль всех посчитанных и непосчитанных параметров
        unknown_parameters=[]
        for att in self.list_of_parameters:
            val=getattr(self,'_'+att)
            if not isinstance(val, str):
                if self.exist(val):
                    if att in self.init_data:
                        init_data='*'
                    else:
                        init_data=''
                    print (init_data,att,' = ', val)
                else:
                    unknown_parameters.append(att)
        if len(unknown_parameters)>0:
            print('Unknown parameters:')
        for att in unknown_parameters:
            print (att,' = ', getattr(self,'_'+att))
            
    def calculate(self): #пытаемся посчиать все параметры
        for key in self.__dict__.keys():
            if key != '_name':
                _key=key[1:]
                if key[0] == '_' and _key not in self.init_data:
                    getattr(self,_key)        
            
class CrossSection(IsentropicFlow):
    #смысл этого класса в том, что он разрешает процессы, происходящие в неком абстрактном сечении потока. Здесь задаются параметры: полные параметры потока, статические, статическое "заднее" давление, которое равно обычному статическому при дозвуковом перепаде и не равно ему при сверхзвуковом, расход смеси газов, площадь сечения, скорость в сечении
    list_of_parameters=IsentropicFlow.list_of_parameters+['Ps_back','flowdensity_thru_Ps','flowdensity_thru_G','flowdensity_error','G','G_corr','G_corr_s','capacity','capacity_s',
                                                          'F','V','V_corr','M','W','Tref','Pref','Force','Momentum','D_hydraulic','wetted_perimeter','Re','dynamic_viscosity',
                                                          'kinematic_viscosity','Re_s','dynamic_viscosity_s','kinematic_viscosity_s','therm_cond','Pr','therm_cond_s','Pr_s']
    #!!!НУЖНО ПОРАБОТАТЬ НАД ОПТИМИЗАЦИЕЙ ЭТОГО КЛАССА, ТК ОН ОДИН ИЗ БАЗОВЫХ И К НЕМУ ИДЕТ СОТНИ И ТЫСЯЧИ ОБРАЩЕНИЙ
    #TODO!!! вынести в отдельную функцию проверку того, является ли имеющийся расход воздуха через сечеине сверхкритическим или докритическим
    def __init__(self,**parameters):
        #СНАЧАЛА ПЕРЕЧИСОЯЕМ ВСЕ ПАРАМЕТРЫ ИСПОЛЬЗУЕМЫЕ ДЛЯ ОПИСАНИЕ СОСТОЯНИЯ В СЕЧЕНИИ ПОТОКА
        super().__init__(**parameters)
        #TODO!! попробовать использовать вместо скалярных значений массив ndarray c одним значением, это нужно для того, чтобы была возможность передавать значения по ссылке. По тестам если использовать массив с одним числом вместо скаляра на быстродействии это скажется только по части записи, чтение по скорости не отдичается. Зато передача параметра по ссылке может придать ускорение в други
        # 1 группа параметров зависящих только от состава смеси
        # self._mass_comp=mass_comp
        # if self.exist(self._mass_comp):
        #     self.init_data.add('mass_comp')
        # self._name = name #имя родительского узла, необходимо, чтобы при возникновении ошибки выводить в лог это имя
        self._Ps_back=self.get_init_data('Ps_back')#фактически имеющееся статическое давление на выходе из сечения, т.е. оно не привязано к параметрам перед сечением
        self._flowdensity_thru_Ps=np.nan #плотность потока вычисляемая через статическое давление на выходе, она может быть больше критической, что нефизично, это нужно контролировать
        self._flowdensity_thru_G=np.nan #плотность потока вычисляемая через заданный физический расход, она может быть больше критической, что нефизично, это нужно контролировать
        self._flowdensity_error=np.nan #невязка по плотности потока, нужна например тогда, когда мы точно знаем, что пототк в сечении не может быть сверхзвуковым, в этом случае, чтобы эта невязка была равна 0 - подробнее смотри алгоритм ее вычисления
        # 6 расходные/скоростные параметры, площадь сечения
        self._G=self.get_init_data('G')
        self._G_corr=self.get_init_data('G_corr') #приведенный расход
        self._G_corr_s = self.get_init_data('G_corr_s') #приведенный расход по статическим параметрам
        self._capacity=self.get_init_data('capacity') #пропускная способность вычисляемая на основе полных параметров, здесь не учитывается эффект запирания потока, если он есть, просто G*T**0.5/P
        self._capacity_s=self.get_init_data('capacity_s') #пропускная способность по статическим параметрам, возможное запирание не учитывается
        # self.flowdensity_thru_Ps_back=np.nan #плотность тока - отношение расхода к площади поперечного сечения
        
        self._F=self.get_init_data('F') #площадь сечения
        self._wetted_perimeter=self.get_init_data('wetted_perimeter') #смоченный периметр для вычисления гидравлического диаметра
        self._V_corr=self.get_init_data('V_corr') #безразмертная скорость лямбда
        self._M=self.get_init_data('M') #число Маха
        self._V=self.get_init_data('V') #скорость
        self._W=self.get_init_data('W') #тепловой поток (G*H)
        # 7 группа
        self._D_hydraulic = self.get_init_data('D_hydraulic') #//гидравлический диаметр, необходим прежде всего для вычисления числа Рейнольдса
        self._Re=np.nan #//Рейнольдс!!!Вычисляется по полным параметрам
        self._Re_s=np.nan #//Рейнольдс !!!Вычисляется по статическим параметрам
        self._dynamic_viscosity = np.nan #//динамическая вязкость. !!!Вычисляется по полным параметрам
        self._dynamic_viscosity_s = np.nan #//динамическая вязкость. !!!Вычисляется по статическим параметрам
        self._kinematic_viscosity = np.nan #//кинематическая вязкость !!!Вычисляется по полным параметрам
        self._kinematic_viscosity_s = np.nan #//кинематическая вязкость !!!Вычисляется по статическим параметрам
        self._therm_cond=self.get_init_data('therm_cond')
        self._Pr = self.get_init_data('Pr')
        self._therm_cond_s=self.get_init_data('therm_cond_s')
        self._Pr_s = self.get_init_data('Pr_s')
        # 8 группа
        self._Tref = 288.15 #температура для расчета приведенных параметров
        self.init_data.add('Tref')
        self._Pref = 101325.0 #давление для расчета приведенных параметров
        self.init_data.add('Pref')
        # 9 параметры импульса и силы от давления в сечении
        self._Force=self.get_init_data('Force')
        self._Momentum=self.get_init_data('Momentum')
        
    @property
    def Re_s(self):
        if self.not_exist(self._Re_s):
            self.calc_now.add('Re_s')
            if self.check_calc_now('kinematic_viscosity_s','V','D_hydraulic') and self.exist(self.V,self.D_hydraulic,self.kinematic_viscosity_s):
                self._Re_s=self.V*self.D_hydraulic/self.kinematic_viscosity_s
            self.calc_now.remove('Re_s')
        self.show_error(self._Re_s,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: Re_s ='.format(parent=self._name))
        return self._Re_s
    @Re_s.setter
    def Re_s(self,x):
        self._Re_s=x
        if np.isnan(x):
            self.init_data.discard('Re_s')
        else:
            self.init_data.add('Re_s') 
            
    @property
    def Re(self):
        if self.not_exist(self._Re):
            self.calc_now.add('Re')
            if self.check_calc_now('kinematic_viscosity','V','D_hydraulic') and self.exist(self.V,self.D_hydraulic,self.kinematic_viscosity):
                self._Re=self.V*self.D_hydraulic/self.kinematic_viscosity
            self.calc_now.remove('Re')
        self.show_error(self._Re,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: Re ='.format(parent=self._name))
        return self._Re
    @Re.setter
    def Re(self,x):
        self._Re=x
        if np.isnan(x):
            self.init_data.discard('Re')
        else:
            self.init_data.add('Re') 

    @property
    def kinematic_viscosity_s(self):
        if self.not_exist(self._kinematic_viscosity_s):
            self.calc_now.add('kinematic_viscosity_s')
            if self.check_calc_now('Ros','dynamic_viscosity_s') and self.exist(self.Ros,self.dynamic_viscosity_s):
                self._kinematic_viscosity_s=self.dynamic_viscosity_s/self.Ros
            self.calc_now.remove('kinematic_viscosity_s')
        self.show_error(self._kinematic_viscosity_s,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: kinematic_viscosity_s ='.format(parent=self._name))
        return self._kinematic_viscosity_s
    @kinematic_viscosity_s.setter
    def kinematic_viscosity_s(self,x):
        self._kinematic_viscosity_s=x
        if np.isnan(x):
            self.init_data.discard('kinematic_viscosity_s')
        else:
            self.init_data.add('kinematic_viscosity_s') 
            
    @property
    def kinematic_viscosity(self):
        if self.not_exist(self._kinematic_viscosity):
            self.calc_now.add('kinematic_viscosity')
            if self.check_calc_now('Ro','dynamic_viscosity') and self.exist(self.Ro,self.dynamic_viscosity):
                self._kinematic_viscosity=self.dynamic_viscosity/self.Ro
            self.calc_now.remove('kinematic_viscosity')
        self.show_error(self._kinematic_viscosity,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: kinematic_viscosity ='.format(parent=self._name))
        return self._kinematic_viscosity
    @kinematic_viscosity.setter
    def kinematic_viscosity(self,x):
        self._kinematic_viscosity=x
        if np.isnan(x):
            self.init_data.discard('kinematic_viscosity')
        else:
            self.init_data.add('kinematic_viscosity') 

    #TODO!!! нужно сделать какуюто настройку, чтобы можно было менять формулу по которой считаертся вязкость
    @property
    def dynamic_viscosity_s(self):
        if self.not_exist(self._dynamic_viscosity_s):
            self.calc_now.add('dynamic_viscosity_s')
            if self.check_calc_now('Ts','mass_comp') and self.exist(self.Ts) and self.exist(self.mass_comp):
                self._dynamic_viscosity_s=Dyn_viscosity(self.Ts,self.mass_comp)
            self.calc_now.remove('dynamic_viscosity_s')
        self.show_error(self._dynamic_viscosity_s,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: dynamic_viscosity_s ='.format(parent=self._name))
        return self._dynamic_viscosity_s
    @dynamic_viscosity_s.setter
    def dynamic_viscosity_s(self,x):
        self._dynamic_viscosity_s=x
        if np.isnan(x):
            self.init_data.discard('dynamic_viscosity_s')
        else:
            self.init_data.add('dynamic_viscosity_s')    
            
    #TODO!!! нужно сделать какуюто настройку, чтобы можно было менять формулу по которой считаертся вязкость
    @property
    def dynamic_viscosity(self):
        if self.not_exist(self._dynamic_viscosity):
            self.calc_now.add('dynamic_viscosity')
            if self.check_calc_now('T','mass_comp') and self.exist(self.T) and self.exist(self.mass_comp):
                self._dynamic_viscosity=Dyn_viscosity(self.T,self.mass_comp)
            self.calc_now.remove('dynamic_viscosity')
        self.show_error(self._dynamic_viscosity,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: dynamic_viscosity ='.format(parent=self._name))
        return self._dynamic_viscosity
    @dynamic_viscosity.setter
    def dynamic_viscosity(self,x):
        self._dynamic_viscosity=x
        if np.isnan(x):
            self.init_data.discard('dynamic_viscosity')
        else:
            self.init_data.add('dynamic_viscosity')   

    @property
    def D_hydraulic(self):
        if self.not_exist(self._D_hydraulic):
            self.calc_now.add('D_hydraulic')
            if self.check_calc_now('F','wetted_perimeter') and self.exist(self.F,self.wetted_perimeter):
                self._D_hydraulic=4*self.F/self.wetted_perimeter
            self.calc_now.remove('D_hydraulic')
        self.show_error(self._D_hydraulic,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: D_hydraulic ='.format(parent=self._name))
        return self._D_hydraulic
    @D_hydraulic.setter
    def D_hydraulic(self,x):
        self._D_hydraulic=x
        if np.isnan(x):
            self.init_data.discard('D_hydraulic')
        else:
            self.init_data.add('D_hydraulic')

    @property
    def wetted_perimeter(self):
        self.show_error(self._wetted_perimeter,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: wetted_perimeter ='.format(parent=self._name))
        return self._wetted_perimeter
    @wetted_perimeter.setter
    def wetted_perimeter(self,x):
        self._wetted_perimeter=x
        if np.isnan(x):
            self.init_data.discard('wetted_perimeter')
        else:
            self.init_data.add('wetted_perimeter')
        
    @property
    def Ps_back(self):
        self.show_error(self._Ps_back,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: Ps_back ='.format(parent=self._name))
        return self._Ps_back
    @Ps_back.setter
    def Ps_back(self,x):
        self._Ps_back=x
        if np.isnan(x):
            self.init_data.discard('Ps_back')
        else:
            self.init_data.add('Ps_back')
            
    @property
    def Ps(self):
        if self.not_exist(self._Ps):
            self.calc_now.add('Ps')
            TsPT= self.check_calc_now('P','T','Ts') and self.exist(self.P, self.T,self.R, self.Ts) and self.exist(self.mass_comp)
            TsRos=self.check_calc_now('Ros','Ts') and self.exist(self.Ros, self.R, self.Ts)
            Psback=self.exist(self.Ps_back)
            if int(TsPT) + int(TsRos) + int(Psback)>1:
                solverLog.warning('WARNING! ' + self._name + ' Overdetermined source data for calculating of Ps.')
                raise SystemExit
            if TsPT: #ищем давление через энтропию, дельта энтропии = 0
                if self.Ts>=self.T:
                    #TODO! костыль ниже!
                    if self.Ts-self.T<0.000001:
                        self.Ts=self.T
                    else:
                        solverLog.error('Error! ' + self._name + ': calculating Ps: Ts({Ts})>T({T})'.format(T=self.T,Ts=self.Ts)) 
                        raise SystemExit
                self._Ps = P2_thru_P1T1T2(self.P, self.T, self.Ts, self.mass_comp,self.R)
                # self.was_edited = True
            elif TsRos: #ищем давление из P/Ro=R*T
                self._Ps = self.R*self.Ts*self.Ros
                # self.was_edited = True
            elif Psback:
                if self.exist(self.Ps_cr):
                    if self.Ps_cr>self.Ps_back:
                        self._Ps=self.Ps_cr
                    else:
                        self._Ps=self.Ps_back
            self.calc_now.remove('Ps')
        self.show_error(self._Ps,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Ps ='.format(parent=self._name))
        return self._Ps
    @Ps.setter
    def Ps(self,x):
        self._Ps=x
        if np.isnan(x):
            self.init_data.discard('Ps')
        else:
            self.init_data.add('Ps')
            
    @property
    def Ts(self):
        if self.not_exist(self._Ts):
            self.calc_now.add('Ts')
            RosPs=self.check_calc_now('Ros','Ps') and self.exist(self.Ros, self.Ps, self.R)
            TPPs=self.check_calc_now('T','P','Ps') and self.exist(self.T, self.P, self.R, self.Ps) and self.exist(self.mass_comp)
            Hs=self.check_calc_now('Hs') and self.exist(self.Hs) and self.exist(self.mass_comp)
            GPTF=self.check_calc_now('G','P','T','H','F','Ts_cr') and self.exist(self.G,self.P,self.T,self.H,self.F,self.Ts_cr,self.R)  and self.exist(self.mass_comp)
            HM=self.check_calc_now('H','M') and self.exist(self.H,self.M) and self.exist(self.mass_comp)
            if int(RosPs) + int(TPPs) + int(Hs) + int(GPTF) + int(HM) >1:
                solverLog.warning('WARNING! CrossSection obj: ' + self._name + f' Calculating Ts: Overdetermined source data. RosPs={RosPs}, TPPs={TPPs}, Hs={Hs}, GPTF={GPTF}, HM={HM}')
            if RosPs:
                self._Ts = self.Ps / self.Ros / self.R
                # self.was_edited = True
            elif TPPs:
                if self.Ps>=self.P:
                    solverLog.error('ERROR! ' + self._name + ': calculate_Ts: Ps({Ps})>P({P})'.format(P=str(self.P),Ps=str(self.Ps)))         
                self._Ts=T2_thru_P1T1P2(self.P,self.T,self.Ps,self.mass_comp,self.R,150,self.T)
                # self.was_edited = True
            elif Hs:
                self._Ts = T_thru_H(self.Hs, self.mass_comp,150,self.T)
                # self.was_edited = True      
            elif GPTF:
                self._Ts=Ts_thru_GPTHF(self.G,self.P,self.T,self.H,self.F,self.Ts_cr,self.mass_comp,self.R)
            elif HM:
                if self.M>2:
                    solverLog.warning(f'WARNING! При вычислении статической температуры в объекте CrossSection name={self.name} обнаружено большое число Маха = {self.M}. Возможна ошибка в дальнейшем, т.к. для метода поиска температуры Ts_thru_HM задан нижний предел 180К')
                self._Ts=Ts_thru_HM(self.H,self.M,self.mass_comp,self.R,180,self.T)
            self.calc_now.remove('Ts')
        self.show_error(self._Ts,text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Ts ='.format(parent=self._name))
        return self._Ts
    @Ts.setter
    def Ts(self,x):
        self._Ts=x
        if np.isnan(x):
            self.init_data.discard('Ts')
        else:
            self.init_data.add('Ts')
        
    @property
    def G(self):
        if self.not_exist(self._G):
            self.calc_now.add('G')
            RosVF=self.check_calc_now('Ros','V', 'F') and self.exist(self.Ros,self.V,self.F)
            G_corrTP=self.check_calc_now('G_corr','T', 'P') and self.exist(self.G_corr,self.T,self.P,self.Tref,self.Pref)
            CapTP=self.check_calc_now('capacity','T', 'P') and self.exist(self.capacity,self.T,self.P)
            if int(RosVF) + int(G_corrTP) + int(CapTP)>1:
                solverLog.warning('WARNING! ' + self._name + ' Calculating G: Overdetermined source data')
                print('overdetermined G')
            if G_corrTP:
                self._G=self.G_corr/(np.sqrt(self.T/self.Tref)*self.Pref/self.P)
                # self.was_edited = True
            elif CapTP:
                self._G=self.capacity/(np.sqrt(self.T))*self.P
                # self.was_edited = True
            elif RosVF:
                self._G = self.Ros*self.V*self.F
                # self.was_edited = True
            self.calc_now.remove('G')
        self.show_error(self._G,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: G ='.format(parent=self._name))
        return self._G
    @G.setter
    def G(self,x):
        self._G=x
        if np.isnan(x):
            self.init_data.discard('G')
        else:
            self.init_data.add('G')

    @property
    def G_corr(self):
        if self.not_exist(self._G_corr):
            self.calc_now.add('G_corr')
            if self.check_calc_now('capacity') and self.exist(self.capacity) and self.exist(self.Pref) and self.exist(self.Tref):
                self._G_corr = self.capacity*self.Pref / np.sqrt(self.Tref)
            self.calc_now.remove('G_corr')
        self.show_error(self._G_corr,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: G_corr ='.format(parent=self._name))
        return self._G_corr
    @G_corr.setter
    def G_corr(self,x):
        self._G_corr=x
        if np.isnan(x):
            self.init_data.discard('G_corr')
        else:
            self.init_data.add('G_corr')

    @property
    def G_corr_s(self):
        if self.not_exist(self._G_corr_s):
            self.calc_now.add('G_corr_s')
            if self.check_calc_now('capacity') and self.exist(self.capacity) and self.exist(self.Pref) and self.exist(self.Tref):
                self._G_corr_s = self.capacity_s*self.Pref / np.sqrt(self.Tref)
            self.calc_now.remove('G_corr_s')
        self.show_error(self._G_corr_s,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: G_corr_s ='.format(parent=self._name))
        return self._G_corr_s
    @G_corr_s.setter
    def G_corr_s(self,x):
        self._G_corr_s=x
        if np.isnan(x):
            self.init_data.discard('G_corr_s')
        else:
            self.init_data.add('G_corr_s')

    @property
    def capacity(self):
        if self.not_exist(self._capacity):
            self.calc_now.add('capacity')
            if self.check_calc_now('G','P','T') and self.exist(self.G) and self.exist(self.P) and self.exist(self.T):
                self._capacity = self.G*np.sqrt(self.T) / self.P
            self.calc_now.remove('capacity')
        self.show_error(self._capacity,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: capacity ='.format(parent=self._name))
        return self._capacity
    @capacity.setter
    def capacity(self,x):
        self._capacity=x
        if np.isnan(x):
            self.init_data.discard('capacity')
        else:
            self.init_data.add('capacity')
            
    @property
    def capacity_s(self):
        if self.not_exist(self._capacity_s):
            self.calc_now.add('capacity_s')
            if self.check_calc_now('G','Ps','Ts') and self.exist(self.G) and self.exist(self.Ps) and self.exist(self.Ts):
                self._capacity_s = self.G*np.sqrt(self.Ts) / self.Ps
            self.calc_now.remove('capacity_s')
        self.show_error(self._capacity_s,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: capacity_s ='.format(parent=self._name))
        return self._capacity_s
    @capacity_s.setter
    def capacity_s(self,x):
        self._capacity_s=x
        if np.isnan(x):
            self.init_data.discard('capacity_s')
        else:
            self.init_data.add('capacity_s')
                     
    @property
    def flowdensity_thru_G(self):
        if self.not_exist(self._flowdensity_thru_G):
            self.calc_now.add('flowdensity_thru_G')
            if self.check_calc_now('G','F') and self.exist(self.G) and self.exist(self.F):
                self._flowdensity_thru_G=self.G/self.F
            self.calc_now.remove('flowdensity_thru_G')
        self.show_error(self._flowdensity_thru_G,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: flowdensity_thru_G ='.format(parent=self._name))
        return self._flowdensity_thru_G
    @flowdensity_thru_G.setter
    def flowdensity_thru_G(self,x):
        self._flowdensity_thru_G=x
        if np.isnan(x):
            self.init_data.discard('flowdensity_thru_G')
        else:
            self.init_data.add('flowdensity_thru_G')

    @property
    def flowdensity_thru_Ps(self):
        if self.not_exist(self._flowdensity_thru_Ps):
            self.calc_now.add('flowdensity_thru_Ps')
            if self.check_calc_now('flowdensity') and self.exist(self.flowdensity):
                self._flowdensity_thru_Ps=self.flowdensity
            self.calc_now.remove('flowdensity_thru_Ps')
        self.show_error(self._flowdensity_thru_Ps,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: flowdensity_thru_Ps ='.format(parent=self._name))
        return self._flowdensity_thru_Ps
    @flowdensity_thru_Ps.setter
    def flowdensity_thru_Ps(self,x):
        self._flowdensity_thru_Ps=x
        if np.isnan(x):
            self.init_data.discard('flowdensity_thru_Ps')
        else:
            self.init_data.add('flowdensity_thru_Ps')
            
    #невязка по плотности потока, нужна например тогда, когда мы точно знаем, что пототк в сечении не может быть сверхзвуковым, в этом случае, чтобы эта невязка была равна 0 - подробнее смотри алгоритм ее вычисления
    #глобально это свойство нужно например при расчете сопла, когда заранее неизвестно докритика на сопле или сверхкритика
    @property
    def flowdensity_error(self):
        if self.not_exist(self._flowdensity_error):
            if self.check_calc_now('flowdensity_thru_G','flowdensity_cr','Ps_back','Ps_cr') and self.exist(self.flowdensity_thru_G,self.flowdensity_cr,self.Ps_back,self.Ps_cr):
                if self.flowdensity_cr<self.flowdensity_thru_G:
                    self._flowdensity_error=(self.flowdensity_thru_G-self.flowdensity_cr)/self.flowdensity_thru_G
                else:
                    if self.Ps_back<self.Ps_cr:
                        self._Ps=self.Ps_cr
                    else:
                        self._Ps=self.Ps_back
                    self._flowdensity_error=(self.flowdensity_thru_G-self.flowdensity_thru_Ps)/self.flowdensity_thru_G
        self.show_error(self._flowdensity_error,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: flowdensity_error ='.format(parent=self._name))
        return self._flowdensity_error
    @flowdensity_error.setter
    def flowdensity_error(self,x,init_data=False):
        self._flowdensity_error=x
        # if np.isnan(x):
        #     self.init_data.discard('flowdensity_error')
        # else:
        #     self.init_data.add('flowdensity_error')
           
    @property
    def F(self):
        if self.not_exist(self._F):
            self.calc_now.add('F')
            if self.check_calc_now('G','Ros','V') and self.exist(self.G,self.Ros,self.V):
                self.F = self.G / self.Ros / self.V
            self.calc_now.remove('F')
        self.show_error(self._F,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: F ='.format(parent=self._name))
        return self._F
    @F.setter
    def F(self,x):
        self._F=x
        if np.isnan(x):
            self.init_data.discard('F')
        else:
            self.init_data.add('F')
    
    @property
    def V(self):
        if self.not_exist(self._V):
            self.calc_now.add('V')
            #TODO! бывают ситуации, когда параметр можно вычислить разными способами, т.е. есть различные наборы данных, позволяющих вычислить этот параметр, нужно как-то это учесть. Как - пока не придумал:(
            GRosF=self.check_calc_now('G','Ros','F','H','Hs') and self.exist(self.G,self.Ros,self.F) and self.not_exist(self.H, self.Hs) #and np.isnan(self.Ps) and np.isnan(self.Ts) 
            VcrVcorr=self.check_calc_now('V_cr','V_corr') and self.exist(self.V_cr,self.V_corr) #and not(np.isnan(self.R)) and not(np.all(np.isnan(self.mass_comp)))
            # MVsnd=self.check_calc_now('M','Vsnd') and self.exist(self.M,self.Vsnd)
            HHs=self.check_calc_now('H','Hs') and self.exist(self.H, self.Hs)
            if int(GRosF) + int(VcrVcorr)+int(HHs)>1: #+int(MVsnd)
                solverLog.warning('WARNING! ' + self._name + ' Calculating V: Overdetermined source data')
                print('!!!!!!!!!! overdetermined V. GRosF={GRosF} VcrVcorr={VcrVcorr} HHs={HHs}'.format(GRosF=str(GRosF), VcrVcorr=str(VcrVcorr), MVsnd=(MVsnd), HHs=(HHs)))  #MVsnd={MVsnd}
            if VcrVcorr:
                self._V = self.V_cr*self.V_corr
            # elif MVsnd:
            #     self._V = self.M*self.Vsnd
            elif HHs: #при вычислении через G и F 
                #TODO!!! костыль ниже. Изредка бывает что Hs едва больше чем H - когдса скорсоть равна 0, в этмо случае
                if self.Hs>self.H:
                    self._V=0
                else:
                    self._V = np.sqrt(2*(self.H-self.Hs))
            elif GRosF:
                self._V = self.G / self.Ros / self.F
            # self.was_edited = True
            self.calc_now.remove('V')
        self.show_error(self._V,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: V ='.format(parent=self._name))
        return self._V
    @V.setter
    def V(self,x):
        self._V=x
        if np.isnan(x):
            self.init_data.discard('V')
        else:
            self.init_data.add('V')    

    # def calculate_V_thru_HHs(self):
    #     if (np.isnan(self.V)) and not(np.isnan(self.H)) and not(np.isnan(self.Hs)):
    #         self.V=np.sqrt(2*(self.H-self.Hs))
    #         self.was_edited = True
            
    @property
    def V_corr(self):
        if self.not_exist(self._V_corr):
            self.calc_now.add('V_corr')
            if self.check_calc_now('V_cr','V') and self.exist(self.V_cr,self.V):
                self._V_corr = self.V/self.V_cr
            self.calc_now.remove('V_corr')
        self.show_error(self._V_corr,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: V_corr ='.format(parent=self._name))
        return self._V_corr
    @V_corr.setter
    def V_corr(self,x):
        self._V_corr=x
        if np.isnan(x):
            self.init_data.discard('V_corr')
        else:
            self.init_data.add('V_corr')

    @property
    def M(self):
        if self.not_exist(self._M):
            self.calc_now.add('M')
            if self.check_calc_now('V','Vsnd') and self.exist(self.Vsnd,self.V):
                self._M = self.V / self.Vsnd
            self.calc_now.remove('M')
        self.show_error(self._M,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: M ='.format(parent=self._name))
        return self._M
    @M.setter
    def M(self,x):
        self._M=x
        if np.isnan(x):
            self.init_data.discard('M')
        else:
            self.init_data.add('M')
            
    @property
    def W(self):
        if self.not_exist(self._W):
            self.calc_now.add('W')
            if self.check_calc_now('H','G') and self.exist(self.H,self.G):
                self._W = self.H * self.G
            self.calc_now.remove('W')
        self.show_error(self._W,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: W ='.format(parent=self._name))
        return self._W
    @W.setter
    def W(self,x):
        self._W=x
        if np.isnan(x):
            self.init_data.discard('W')
        else:
            self.init_data.add('W')
            
    @property
    def Tref(self):
        self.show_error(self._Tref,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: Tref ='.format(parent=self._name))
        return self._Tref
    @Tref.setter
    def Tref(self,x):
        self._Tref=x
        if np.isnan(x):
            self.init_data.discard('Tref')
        else:
            self.init_data.add('Tref')
            
    @property
    def Pref(self):
        self.show_error(self._Pref,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: Pref ='.format(parent=self._name))
        return self._Pref
    @Pref.setter
    def Pref(self,x):
        self._Pref=x
        if np.isnan(x):
            self.init_data.discard('Pref')
        else:
            self.init_data.add('Pref')
            
    @property
    def Force(self):
        if self.not_exist(self._Force):
            self.calc_now.add('Force')
            if self.check_calc_now('Ps','F') and self.exist(self.Ps,self.F):
                self._Force = self.Ps*self.F
            self.calc_now.remove('Force')
        self.show_error(self._Force,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: Force ='.format(parent=self._name))
        return self._Force
    @Force.setter
    def Force(self,x):
        self._Force=x
        if np.isnan(x):
            self.init_data.add('Force')
        else:
            self.init_data.add('Force')
            
    @property
    def Momentum(self):
        if self.not_exist(self._Momentum):
            self.calc_now.add('Momentum')
            if self.check_calc_now('G','V') and self.exist(self.G,self.V):
                self._Momentum = self.G*self.V
            self.calc_now.remove('Momentum')
        self.show_error(self._Momentum,text_to_show='ERROR! CrossSection obj: {parent}: Noncalculable variable: Momentum ='.format(parent=self._name))
        return self._Momentum
    @Momentum.setter
    def Momentum(self,x):
        self._Momentum=x
        if np.isnan(x):
            self.init_data.discard('Momentum')
        else:
            self.init_data.add('Momentum')

    @property
    def therm_cond(self):
        if self.not_exist(self._therm_cond):
            self.calc_now.add('therm_cond')
            if self.check_calc_now('T') and self.exist(self.T):
                self._therm_cond = thermal_conductivity_air_hydra(self.T)
                # self.was_edited = True
            self.calc_now.remove('therm_cond')
        self.show_error(self._therm_cond,
                        text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: therm_cond ='.format(
                            parent=self._name))
        return self._therm_cond

    @therm_cond.setter
    def therm_cond(self, x):
        self.therm_cond = x
        if np.isnan(x):
            self.init_data.discard('therm_cond')
        else:
            self.init_data.add('therm_cond')

    @property
    def Pr(self):
        if self.not_exist(self._Pr):
            self.calc_now.add('Pr')
            if self.check_calc_now('T') and self.exist(self.T):
                self._Pr = Prandtl_hydra(self.T)
                # self.was_edited = True
            self.calc_now.remove('Pr')
        self.show_error(self._Pr,
                        text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Pr ='.format(
                            parent=self._name))
        return self._Pr

    @Pr.setter
    def Pr(self, x):
        self._Pr = x
        if np.isnan(x):
            self.init_data.discard('Pr')
        else:
            self.init_data.add('Pr')

    @property
    def therm_cond_s(self):
        if self.not_exist(self._therm_cond_s):
            self.calc_now.add('therm_cond_s')
            if self.check_calc_now('Ts') and self.exist(self.Ts):
                self._therm_cond = thermal_conductivity_air_hydra(self.Ts)
                # self.was_edited = True
            self.calc_now.remove('therm_cond_s')
        self.show_error(self._therm_cond_s,
                        text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: therm_cond_s ='.format(
                            parent=self._name))
        return self._therm_cond_s

    @therm_cond_s.setter
    def therm_cond_s(self, x):
        self.therm_cond_s = x
        if np.isnan(x):
            self.init_data.discard('flowdensity_cr')
        else:
            self.init_data.add('flowdensity_cr')

    @property
    def Pr_s(self):
        if self.not_exist(self._Pr_s):
            self.calc_now.add('Pr_s')
            if self.check_calc_now('Ts') and self.exist(self.Ts):
                self._Pr_s = Prandtl_hydra(self.Ts)
                # self.was_edited = True
            self.calc_now.remove('Pr_s')
        self.show_error(self._Pr_s,
                        text_to_show='ERROR! Isentropic Flow obj: {parent}: Noncalculable variable: Pr_s ='.format(
                            parent=self._name))
        return self._Pr_s

    @Pr_s.setter
    def Pr_s(self, x):
        self._Pr_s = x
        if np.isnan(x):
            self.init_data.discard('Pr_s')
        else:
            self.init_data.add('Pr_s')
            
    
    
    # def is_total_parameters_exist(self):
    #     return (self.exist(self.P) or self.exist(self.Ro)) and (self.exist(self.T) or self.exist(self.H))
    
    # def is_static_parameters_exist(self):
    #     return (self.exist(self.Ps) or self.exist(self.Ros)) and (self.exist(self.Ts) or self.exist(self.Hs))
    
    # def is_velocity_exist(self):
    #     return (self.exist(self.V) or self.exist(self.M) or self.exist(self.V_corr))
    
    def calculate(self): #пытаемся посчиать все параметры
        if self.exist(self._F,self._G) and self.not_exist(self._Ps_back): #расчет через площадь сечения и расход, здесь допустим только расчет докритики, при сверхкритике выдаст ошибку
            if self.exist(self.flowdensity_cr):
                if self.G/self.F >self.flowdensity_cr:
                    solverLog.error('ERROR! ' + self.name + ': calculate thru Area and Massflow: Given massflow cant flow through given area - choked flow. G={G} F={F} P={P} T={T}'.format(G=self.G,F=self.F,P=self.P,T=self.T) )
                    # raise SystemExit
                    return (False, self.flowdensity_cr*self.F) #если сечение не пропускает заданный поток, то возвращаем ошибку в лог и кортеж, где первое значение - False, второе - максимально пропускаемый расход
                elif self.exist(self.P,self.T,self.H,self.Ts_cr,self.R) and self.exist(self.mass_comp):
                    self._Ts=Ts_thru_GPTHF(self.G,self.P,self.T,self.H,self.F,self.Ts_cr,self.mass_comp,self.R)
        elif self.exist(self._F,self._Ps_back) and self.not_exist(self._G): #расчет через площадь и статическое давление на выходе, здесь допустима и сверхкритика и докритика, но при сверхкритике расход запирается по критическому статическоум давлению
            if self.exist(self.Ps_cr):
                if self.Ps_cr>self.Ps_back:
                    self._Ps=self.Ps_cr
                else:
                    self._Ps=self.Ps_back
        elif self.exist(self._F,self._G,self._Ps_back): #расчет через площадь, статическое давление на выходе и расход. Цель расчета получить невязку через давление и через расход. Расчет считается сошедшимся тогда, когда невязка равна 0
            self.flowdensity_error #в геттер этого параметра уже заложен нужный алгоритм       
        for key in self.__dict__.keys():
            if key != '_name':
                _key=key[1:]
                if key[0] == '_' and _key not in self.init_data:
                    getattr(self,_key)   
                    
    # def reset_results(self):
    #     for key in self.__dict__.keys():
    #         _key=key[1:]
    #         if key !='_name' and key[0] == '_' and _key not in self.init_data:
    #             if _key=='mass_comp':
    #                 self._mass_comp.fill(np.nan)
    #             else:
    #                 setattr(self,key,np.nan)
                    
                       
    def copy_attributes(self,other_crosssection): #эта штука нужна для копирования значений элементов, т.к. использование простого присваивания ломает пргргамму изза особенностей языка - в Питоне все переменные - объекты, и они все передаются по ссылке (долго объяснять)
        for key in self.__dict__.keys():
            _temp=getattr(other_crosssection,key)
            if not (type(_temp) == str):
                if not np.all(np.isnan(_temp)): #NB! эта проверка нужна потому, что был прецендент: два последовательных канала. у первого канала на выходе площадь сечения не задается, а на входе во втором канале задается площадь сечения. Соответственно она становится равной площади на выходе из перврго сечения - так сейчас реализован алгоритм: объект crosssection на выходе из узла соответствует такому же объекту на входе в следующий узел. Но в процессе расчета параметров внутри первого узла (объекта channel) происходит копирование атрибутов self.outlet.copy_attributes(self.outlet_ideal) из outlet_ideal в outlet, из-за этого параметр F в outlet затирается
                    setattr(self, key, _temp)
            else:
                setattr(self, key, _temp)    
            


# dry_air_test=np.array([7.5512e-01, 2.3150e-01, 1.2920e-02, 4.6000e-04, 0.0000e+00, 0.0000e+00,0.0000e+00])
# A=IsentropicFlow(name='test',mass_comp=dry_air_test,P=200000,T=500)
# A.mass_comp=dry_air_test
# A.P=200000
# A.T=500
# A.Ts=450
# A.status()
# A.reset_results()
        
# B=CrossSection(name='test',mass_comp=dry_air_test,P=200000,T=500,F=0.5)
# B.Ps_back=160000
# # B.G=50
# B.calculate()
# B.status()
# B.reset_results()
# B.status()




# test=CrossSection()
# test.Ps=400000
# test.Ts=400
# test.V_corr=1.5
# # test.P=500000
# # test.T=1000
# dry_air_test=np.array([7.5512e-01, 2.3150e-01, 1.2920e-02, 4.6000e-04, 0.0000e+00, 0.0000e+00,0.0000e+00])
# test.mass_comp=dry_air_test
# # test.F=0.01
# # test.G=6.3
# # # test.Ps=400000
# # # test.Ps_back=200000
# # test.V=500
# # # test.status()
# # test.calculate_thru_FGPback()
# test.calculate_totPar_thru_statParV()
# test.status()

# base=test
# def _test(G,Ps_back):
#     test=copy.deepcopy(base)
#     test.G=G   
#     test.Ps_back=Ps_back
#     test.calculate_thru_FGPback()
#     # test.status()    
#     return test.flowdensity_error

# # # _test(5,300000)



# def _test2(P):
#     test=copy.deepcopy(base)
#     # test.Ps_back=P
#     GGG = optimize.brentq(_test,0.5,20,args=(P,),disp=True)
#     test.G=GGG
#     test.Ps_back=P
#     test.calculate_thru_FGPback()
#     # base.status()
#     return test
    

# rez=_test2(200000)
# rez.status()

# PPP=np.arange(100000,400000,5000)
# xxx=[]

# for P_val in PPP:
#     rez=_test2(P_val)
#     print(P_val)
#     xxx.append(rez)

# XXX=[]
# YYY=[]
# for val in xxx:
#     YYY.append(val.P)
#     XXX.append(val.G)
    
# plt.plot(PPP,XXX)


# test=IsentropicFlow()
# test.P=500000
# test.T=1000
# dry_air_test=np.array([7.5512e-01, 2.3150e-01, 1.2920e-02, 4.6000e-04, 0.0000e+00, 0.0000e+00,0.0000e+00])
# test.mass_comp=dry_air_test
# # test.Ps=450000
# test.calculate()
# Ps=np.arange(100000.0,499000.0,5000.0 )
# flowdensity=[]
# flowdensity_cr=[]
# for Ps_val in Ps:
#     _test=copy.deepcopy(test)
#     _test.Ps=Ps_val
#     _test.calculate()
#     flowdensity.append(_test.flowdensity)
#     flowdensity_cr.append(_test.flowdensity_cr)

# plt.plot(Ps,flowdensity)
# plt.plot(Ps,flowdensity_cr)

#свойства воздуха и топлива
#dry_air_test=np.array([7.5512e-01, 2.3150e-01, 1.2920e-02, 4.6000e-04, 0.0000e+00, 0.0000e+00,0.0000e+00])
#print(P2_thru_P1T1T2(500000,500,400,dry_air_test,285.74582159886364))
#print(T2_thru_P1T1P2(500000,500,225334.47437230457,dry_air_test,285.74582159886364,100,1000))
#wet_air_test=np.array([0.749164,0.229674,0.012818,0.000456,0.007888,0,0])
#fuel_test=np.array([0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00])
#kerosene_air_stoichiometric_combustion_products=np.array([0.7070278421, 0., 0.01209714975, 0.2010579341, 0.07981707402, 0., 0.]) #для керосина и воздуха с условным составом С12H23.325

#print(H(500,dry_air_test))
#print(T_thru_H(200686.4015157996,dry_air_test,200,3000))

#Hu=(2*_H(298.15,coefsCO2,Runiversal)+3*_H(298.15,coefsH2O,Runiversal))-(2*_H(298.15,coefsCH3,Runiversal)+3.5*_H(298.15,coefsO2,Runiversal))/2
#Hukg=Hu/MolW_CH3
#print(Hu)
#R_mix([7.5512e-01, 2.3150e-01, 1.2920e-02, 4.6000e-04, 0.0000e+00, 0.0000e+00,0.0000e+00,0])
#TenDoubles = ctypes.c_double * 7
#ii = TenDoubles(7.5512e-01, 2.3150e-01, 1.2920e-02, 4.6000e-04, 0.0000e+00, 0.0000e+00, 0.0000e+00)
#
#x=PsTsV_thru_GPTF(1,500000,1000,0.00325383,268497,853,574,dry_air_test,285.74582159886364)
#print(x)
"""
ДАЛЕЕ АЛГОРИТМ ДЛЯ ПРОЕВРКИ ФУНКЦИЙ ГРАИФКАМИ
"""
"""
gas_methan_test=np.array([0.737723,0.151292,0.012622,0.052482,0.045249,0,0,0.000631])
dry_air_test=np.array([7.5512e-01, 2.3150e-01, 1.2920e-02, 4.6000e-04, 0.0000e+00, 0.0000e+00,0.0000e+00,0])
kerosene_air_stoichiometric_combustion_products=np.array([0.7070278421, 0., 0.01209714975, 0.2010579341, 0.07981707402, 0., 0.]) #для керосина и кислорода с условным составом С12H23.325
#функция вычисления теплоемкости, энтальпии и энтропии продуктов сгорания на основе относительного расхода топлива - упрощенные формулы для CFD
def Cp_products(FAR,L,T):
    rez=(FAR*Cp(T,kerosene_air_stoichiometric_combustion_products)*(1+L)+Cp(T,dry_air_test)*(1-L*FAR))/(FAR+1)
    return rez
def H_products(FAR,L,T):
    rez=(FAR*H(T,kerosene_air_stoichiometric_combustion_products)*(1+L)+H(T,dry_air_test)*(1-L*FAR))/(FAR+1)
    return rez
def S_products(FAR,L,T):
    rez=(FAR*Sf(T,kerosene_air_stoichiometric_combustion_products)*(1+L)+Sf(T,dry_air_test)*(1-L*FAR))/(FAR+1)
    return rez
    

print("проверка")
#print(Rel_humidity(0.0038,100000,273,dry_air_test))
#print(H(500,dry_air_test))
#print(T_thru_H(H(500,dry_air_test),dry_air_test,200,6000))
#print(PsTsV_thru_GPTF(1,500000,1000,0.010158711,dry_air_test))
#print(Ts_thru_GPTF(1,500000,1000,0.00158711,dry_air_test))
#print(Ts_thru_GPTF(1,500000,1000,0.000158711,dry_air_test))
#print(Ts_thru_HM(8000,0.1,dry_air_test,200,6000))
#для проверки строим графики
#test_func=lambda Ts,F: 1-P2_thru_P1T1T2(500000,1000,Ts,dry_air_test)/R_mix(dry_air_test)/Ts*np.sqrt(2*(H(1000,dry_air_test)-H(Ts,dry_air_test)))*F
#Tmatrix2=np.arange(200.0,1000.0,10)
#Pmatrix2=[]
#for Ts in Tmatrix2:
#    Pmatrix2.append(test_func(Ts,0.01))
    
Tmatrix=np.arange(200.0,3000.0,10)
#CpmatrixN2=[]
#CpmatrixO2=[]
#CpmatrixCO2=[]
#CpmatrixAr=[]
#CpmatrixH2O=[]
#CpmatrixJetA=[]
#CpmatrixJetALiquid=[]
CpmatrixAir=[]
CpmatrixCombProdAlfa1=[]
CpmatrixCombProd1=[]
CpmatrixCombProd2=[]
CpmatrixCombProd3=[]
CpmatrixCombProd4=[]
CpmatrixCombProd5=[]
#HmatrixN2=[]
#HmatrixO2=[]
#HmatrixCO2=[]
#HmatrixAr=[]
HmatrixH2O=[]
#HmatrixJetA=[]
#HmatrixJetALiquid=[]
HmatrixAir=[]
HmatrixCombProdAlfa1=[]
HmatrixCombProd1=[]
HmatrixCombProd2=[]
HmatrixCombProd3=[]
HmatrixCombProd4=[]
HmatrixCombProd5=[]
#SmatrixN2=[]
#SmatrixO2=[]
#SmatrixCO2=[]
#SmatrixAr=[]
#SmatrixH2O=[]
#SmatrixJetA=[]
#SmatrixJetALiquid=[]
SmatrixAir=[]
SmatrixCombProdAlfa1=[]
SmatrixCombProd1=[]
SmatrixCombProd2=[]
SmatrixCombProd3=[]
SmatrixCombProd4=[]
SmatrixCombProd5=[]
#test=[]
#kmatrix=[]
#Tsmatrix=[]
#Psmatrix=[]
#Psatmatrix=[]
#Psat2matrix=[]
#Tsmatrix2=[]
#Psmatrix2=[]
#Vmatrix2=[]
#WARmatrix_minus15=[]
#WARmatrix_0=[]
#WARmatrix_15=[]
#WARmatrix_30=[]
#WARmatrix_sat=[]
#RelHum_m=[]

for T in Tmatrix:
#    CpmatrixN2.append(_Cp(T,coefsN2,R_N2))
#    CpmatrixO2.append(_Cp(T,coefsO2,R_O2))
#    CpmatrixCO2.append(_Cp(T,coefsCO2,R_CO2))
#    CpmatrixAr.append(_Cp(T,coefsAr,R_Ar))
#    CpmatrixH2O.append(_Cp(T,coefsH2O,R_H2O))
#    CpmatrixJetA.append(_Cp(T,coefsJetA,R_JetA))
#    CpmatrixJetALiquid.append(_Cp(T,coefsJetA,R_JetA))
    CpmatrixAir.append(Cp(T,dry_air_test))
    CpmatrixCombProdAlfa1.append(Cp(T,kerosene_air_stoichiometric_combustion_products))
    CpmatrixCombProd1.append(Cp_products(0,14.731261494963519,T))
    CpmatrixCombProd2.append(Cp_products(0.02,14.731261494963519,T))
    CpmatrixCombProd3.append(Cp_products(0.04,14.731261494963519,T))
    CpmatrixCombProd4.append(Cp_products(0.06,14.731261494963519,T))
    CpmatrixCombProd5.append(Cp_products(0.08,14.731261494963519,T))
#    HmatrixN2.append(_H(T,coefsN2,R_N2))
#    HmatrixO2.append(_H(T,coefsO2,R_O2))
#    HmatrixCO2.append(_H(T,coefsCO2,R_CO2))
#    HmatrixAr.append(_H(T,coefsAr,R_Ar))
    HmatrixH2O.append(_H(T,coefsH2O,R_H2O))
#    HmatrixJetA.append(_H(T,coefsJetA,R_JetA))
#    HmatrixJetALiquid.append(_H(T,coefsJetA,R_JetA))
    HmatrixAir.append(H(T,dry_air_test))
    HmatrixCombProdAlfa1.append(H(T,kerosene_air_stoichiometric_combustion_products))
    HmatrixCombProd1.append(H_products(0,14.731261494963519,T))
    HmatrixCombProd2.append(H_products(0.02,14.731261494963519,T))
    HmatrixCombProd3.append(H_products(0.04,14.731261494963519,T))
    HmatrixCombProd4.append(H_products(0.06,14.731261494963519,T))
    HmatrixCombProd5.append(H_products(0.08,14.731261494963519,T))
#    SmatrixN2.append(_Sf(T,coefsN2,R_N2))
#    SmatrixO2.append(_Sf(T,coefsO2,R_O2))
#    SmatrixCO2.append(_Sf(T,coefsCO2,R_CO2))
#    SmatrixAr.append(_Sf(T,coefsAr,R_Ar))
#    SmatrixH2O.append(_Sf(T,coefsH2O,R_H2O))
#    SmatrixJetA.append(_Sf(T,coefsJetA,R_JetA))
#    SmatrixJetALiquid.append(_Sf(T,coefsJetA,R_JetA))
    SmatrixAir.append(Sf(T,dry_air_test))
    SmatrixCombProdAlfa1.append(Sf(T,kerosene_air_stoichiometric_combustion_products))
    SmatrixCombProd1.append(S_products(0,14.731261494963519,T))
    SmatrixCombProd2.append(S_products(0.02,14.731261494963519,T))
    SmatrixCombProd3.append(S_products(0.04,14.731261494963519,T))
    SmatrixCombProd4.append(S_products(0.06,14.731261494963519,T))
    SmatrixCombProd5.append(S_products(0.08,14.731261494963519,T))
#    test.append((lambda T,coefs,R: _Cp(T,coefs,R)-1)(T,coefsO2,R_O2))
#    kmatrix.append(k(T,dry_air_test))
#    Psmatrix.append(P2_thru_P1T1T2(500000,3001,T,dry_air_test))

#Pmatrix=np.arange(5000,500000,1000)
#for P in Pmatrix:
#    Tsmatrix.append(T2_thru_P1T1P2(500000,3001,P,dry_air_test,200,3001))
#Fmatrix=np.arange(0.0001,0.02,0.0001)
#for F in Fmatrix:
#    rez=PsTsV_thru_GPTF(1,500000,1000,F,dry_air_test)
#    Tsmatrix2.append(rez[1])
#    Psmatrix2.append(rez[0])
#    Vmatrix2.append(rez[2])
#Tmatrix2=np.arange(200.0,372.0,1)
#for T in Tmatrix2:
#    Psatmatrix.append(P_sat_vapour1(T))
#    Psat2matrix.append(P_sat_vapour2(100000,T))
#    RelHummatrix=np.arange(0.0,2.0,0.05)
#    WARmatrix_sat.append(WAR(1, 100000, T, dry_air_test))
#for RH in RelHummatrix:
#    WARmatrix_minus15.append(WAR(RH, 100000, (273.15-15), dry_air_test))
#    WARmatrix_0.append(WAR(RH, 100000, (273.15), dry_air_test))
#    WARmatrix_15.append(WAR(RH, 100000, (273.15+15), dry_air_test))
#    WARmatrix_30.append(WAR(RH, 100000, (273.15+30), dry_air_test))
#WARmatrix=np.arange(0,0.05,0.001)
#for WARx in WARmatrix:
#    RelHum_m.append(Rel_humidity(WARx,100000,300))

#проверка функций на графике
#print(S_Air(1000)-S_Air(200))
fig, axes = plt.subplots(3,1)
fig.set_size_inches(15, 55)

#axes[0].plot(Tmatrix,CpmatrixN2,label='N2')
#axes[0].plot(Tmatrix,CpmatrixO2,label='O2')
#axes[0].plot(Tmatrix,CpmatrixCO2,label='CO2')
#axes[0].plot(Tmatrix,CpmatrixAr,label='Ar')
#axes[0].plot(Tmatrix,CpmatrixH2O,label='H2O')
#axes[0].plot(Tmatrix,CpmatrixJetA,label='JetA')
#axes[0].plot(Tmatrix,CpmatrixJetALiquid,label='JetALiquid')
axes[0].plot(Tmatrix,CpmatrixAir,label='сухой воздух')
axes[0].plot(Tmatrix,CpmatrixCombProdAlfa1,label='продукты стехиометрического горения керосина q=0,0678828')
axes[0].plot(Tmatrix,CpmatrixCombProd1,'--r',label='продукты горения керосина q=0')
axes[0].plot(Tmatrix,CpmatrixCombProd2,'--y',label='продукты горения керосина q=0.02')
axes[0].plot(Tmatrix,CpmatrixCombProd3,'--g',label='продукты горения керосина q=0.04')
axes[0].plot(Tmatrix,CpmatrixCombProd4,'--b',label='продукты горения керосина q=0.06')
axes[0].plot(Tmatrix,CpmatrixCombProd5,'--c',label='продукты горения керосина q=0.08')
axes[0].legend(fontsize=8)
axes[0].set_xlabel('Температура, К')
axes[0].set_ylabel('Теплоемкость, Дж/кг/К')
axes[0].set_title('Теплоемкость')
#axes[1].plot(Tmatrix,HmatrixN2,label='N2')
#axes[1].plot(Tmatrix,HmatrixO2,label='O2')
#axes[1].plot(Tmatrix,HmatrixCO2,label='CO2')
#axes[1].plot(Tmatrix,HmatrixAr,label='Ar')
axes[1].plot(Tmatrix,HmatrixH2O,label='H2O')
#axes[1].plot(Tmatrix,HmatrixJetA,label='JetA')
#axes[1].plot(Tmatrix,HmatrixJetALiquid,label='JetALiquid')
axes[1].plot(Tmatrix,HmatrixAir,label='сухой воздух')
axes[1].plot(Tmatrix,HmatrixCombProdAlfa1,label='продукты стехиометрического горения керосина q=0,0678828')
axes[1].plot(Tmatrix,HmatrixCombProd1,'--r',label='продукты горения керосина q=0')
axes[1].plot(Tmatrix,HmatrixCombProd2,'--y',label='продукты горения керосина q=0.02')
axes[1].plot(Tmatrix,HmatrixCombProd3,'--g',label='продукты горения керосина q=0.04')
axes[1].plot(Tmatrix,HmatrixCombProd4,'--b',label='продукты горения керосина q=0.06')
axes[1].plot(Tmatrix,HmatrixCombProd5,'--c',label='продукты горения керосина q=0.08')
axes[1].legend(fontsize=8)
axes[1].set_xlabel('Температура, К')
axes[1].set_ylabel('Удельная энтальпия, Дж/кг')
axes[1].set_title('Энтальпия')
#axes[2].plot(Tmatrix,SmatrixN2,label='N2')
#axes[2].plot(Tmatrix,SmatrixO2,label='O2')
#axes[2].plot(Tmatrix,SmatrixCO2,label='CO2')
#axes[2].plot(Tmatrix,SmatrixAr,label='Ar')
#axes[2].plot(Tmatrix,SmatrixH2O,label='H2O')
#axes[2].plot(Tmatrix,SmatrixJetA,label='JetA')
#axes[2].plot(Tmatrix,SmatrixJetALiquid,label='JetALiquid')
axes[2].plot(Tmatrix,SmatrixAir,label='сухой воздух')
axes[2].plot(Tmatrix,SmatrixCombProdAlfa1,label='продукты стехиометрического горения керосина q=0,0678828')
axes[2].plot(Tmatrix,SmatrixCombProd1,'--r',label='продукты горения керосина q=0')
axes[2].plot(Tmatrix,SmatrixCombProd2,'--y',label='продукты горения керосина q=0.02')
axes[2].plot(Tmatrix,SmatrixCombProd3,'--g',label='продукты горения керосина q=0.04')
axes[2].plot(Tmatrix,SmatrixCombProd4,'--b',label='продукты горения керосина q=0.06')
axes[2].plot(Tmatrix,SmatrixCombProd5,'--c',label='продукты горения керосина q=0.08')
axes[2].legend(fontsize=8)
axes[2].set_xlabel('Температура, К')
axes[2].set_ylabel('Удельная энтропия, Дж/кг/К')
axes[2].set_title('Энтропия')
#axes[3].plot(Tmatrix,kmatrix)
#axes[3].set_xlabel('Температура, К')
#axes[3].set_ylabel('Коэффициент адиабаты')
#axes[3].set_title('Коэффициент адиабаты')
#axes[4].plot(Tmatrix,Psmatrix)
#axes[4].set_xlabel('Температура, К')
#axes[4].set_ylabel('Pstat')
#axes[4].set_title('Статическое давление по функции P2_thru_P1T1T2')
#axes[5].plot(Pmatrix,Tsmatrix)
#axes[5].set_xlabel('P2, Pa')
#axes[5].set_ylabel('T2, K')
#axes[5].set_title('Статическая температура по функции T2_thru_P1T1P2')
#lin1=axes[6].plot(Fmatrix,Tsmatrix2,label='Ts')
#axes[6].set_xlabel('F, m2')
#axes[6].set_ylabel('Ts, K')
#axes[6].set_title('Проверка функции PsTsV_thru_GPTF')
#axes[6].set_ylim(bottom=800,top=1100)
#axes2=axes[6].twinx()
#axes2.set_ylabel('Ps, Па')
#lin2=axes2.plot(Fmatrix,Psmatrix2,label='Ps',color='red')
#lines=lin1+lin2
#labs = [l.get_label() for l in lines]
##axes2[6].set_xlabel('F, m2')
#axes[6].legend(lines, labs, loc=0)
#axes2.set_ylim(bottom=250000,top=550000)
#axes[7].plot(Tmatrix2,Psatmatrix,label='wikipedia')
#axes[7].plot(Tmatrix2,Psat2matrix,label='GTP')
#axes[7].set_xlabel('Температура, К')
#axes[7].set_ylabel('Psat')
#axes[7].set_title('Давление насыщенных паров')
#axes[7].legend()
#axes[8].plot(RelHummatrix,WARmatrix_minus15,label='-15')
#axes[8].plot(RelHummatrix,WARmatrix_0,label='0')
#axes[8].plot(RelHummatrix,WARmatrix_15,label='15')
#axes[8].plot(RelHummatrix,WARmatrix_30,label='30')
#axes[8].set_xlabel('Относительная влажность')
#axes[8].set_ylabel('Влагосодержание (удельная влажность), кг пара/ кг сухого воздуха')
#axes[8].set_title('Влажность')
#axes[8].legend()
#axes[9].plot(Tmatrix2,WARmatrix_sat)
#axes[9].set_xlabel('Температура, К')
#axes[9].set_ylabel('Влагосодержание (удельная влажность), кг пара/ кг сухого воздуха')
#axes[9].set_title('Зависимость влагосодержания в воздухе со 100% влажностью от температуры')
#axes[9].legend()
#axes[10].plot(WARmatrix,RelHum_m)
#axes[10].set_xlabel('Влагосодержание')
#axes[10].set_ylabel('отн влажность')
#axes[10].set_title('Зависимость отн влажность от Влагосодержания')




"""


# Tx=np.arange(200,1001,5,dtype=float)
# Sy=[]
# Sy2=[]
# for T in Tx:
#     Sy.append(Sf(T,dry_air_test))
#     Sy2.append(Sf(T,dry_air_test))
    
# test=np.polyfit(Sy,Tx,deg=5)
# polyfit=np.poly1d(test)

# Sx=np.arange(6300,7800,5,dtype=float)
# Ty=[]
# tol=[]
# for S in Sx:
#     _T=polyfit(S)
#     Ty.append(_T)
#     tol.append((S-Sf(_T,dry_air_test))/S)
    
# plt.plot(Sx,Ty)
# plt.plot(Sx,tol)
    
# def S_polynom(T):
#     rez=1.63404400e-15*T**5-5.89847533e-11*T**4+8.92203708e-07*T**3-6.81430163e-03*T**2+2.59524639e+01*T-3.92324316e+04
#     return rez


# plt.plot(Tx,Sy)
# plt.plot(Tx,Sy2)

