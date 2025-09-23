#-----------------------------------------------------------------------------#
#=============================================================================#
#-----------------------------------------------------------------------------#

# PME-3480 - Motores de Combustão Interna
# 1D Otto cycle simulator - 2025
# Implementation 1 - Group 01

#-----------------------------------------------------------------------------#
#=============================================================================#
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
#-----------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import OttoCycle as oc  # Nosso módulo principal do simulador

#-----------------------------------------------------------------------------#
#=============================================================================#
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# 2. PARÂMETROS E CONSTANTES DO PROJETO (Matheus)
#-----------------------------------------------------------------------------#
# Parâmetros fixos do motor (Grupo 1)
B = 60 / 1000      # Diâmetro do cilindro (m)
S = 120 / 1000     # Curso do pistão (m)
L = 180 / 1000     # Comprimento da biela (m)
n_rpm = 2000       # Rotação do motor (rpm)
n = n_rpm / 60     # Rotação do motor (rps)
fuel = 'CH4'       # Combustível (Metano)
Zc = 1                 # nº de cilindros (S1)
x  = 2                 # motor 4T -> 2 voltas por ciclo
Vu = (np.pi/4.0) * B**2 * S #Volume util do cilindro
omega = 2.0*np.pi*n    # rad/s
#Vc = Vu/(rv - 1)
phi = 0.7 #razão de equivalência
AFest = (4.76 * 2 * 28.97)/16.043 # razão ar-comb estequiométrica
AFreal = AFest/phi # razão ar-comb real
rho_ar = 1.184 #massa específica de ar na condição de teste

# Parâmetros da combustão (Função de Wiebe para S1)
ThSOC = -10. * (np.pi/180) # Início da combustão (rad)
ThEOC = +10. * (np.pi/180) # Fim da combustão (rad)
aWF = 5.0          # Fator de eficiência de Wiebe
mWF = 2.0          # Fator de forma de Wiebe

# Parâmetros de contorno (ambiente do laboratório)
pint = 100e3       # Pressão de admissão (Pa)
Tint = 273.15 + 25 # Temperatura de admissão (K)
pexh = 100e3       # Pressão de escape (Pa)
#phi = 1.0          # Razão de equivalência

# Constantes físicas
PCI_CH4 = 50.01e6  # Poder Calorífico Inferior do Metano (J/kg)

# Ângulo do virabrequim para a simulação
Th0 = -360. * (np.pi/180)
Th1 = +360. * (np.pi/180)
Ths = 1e-1 * (np.pi/180)
Thn = int(((Th1 - Th0) / Ths) + 1)
Th = np.linspace(start=Th0, stop=Th1, num=Thn, endpoint=True)

#-----------------------------------------------------------------------------#
# 3. LEITURA DOS DADOS EXPERIMENTAIS (Matheus)
#-----------------------------------------------------------------------------#
dados_exp = np.loadtxt('grupo01-PerformanceParameters.txt', skiprows=5, encoding='latin1')
rv_exp = dados_exp[:, 0]
Texh_exp_C = dados_exp[:, 1]
mpF_exp_kg_h = dados_exp[:, 2]
Mt_exp_kgfm = dados_exp[:, 3]

#-----------------------------------------------------------------------------#
#=============================================================================#
#-----------------------------------------------------------------------------#

#-----------------------------------------------------------------------------#
# 4. LOOP PRINCIPAL DE SIMULAÇÃO E CÁLCULOS
#-----------------------------------------------------------------------------#
# Estrutura para armazenar os resultados finais
resultados_finais = []

for i in range(len(rv_exp)):
    # Pega os parâmetros para a simulação atual
    rv = rv_exp[i]
    Vc = Vu/(rv - 1)
    Texh_K = Texh_exp_C[i] + 273.15
    mpF_kg_s = mpF_exp_kg_h[i] / 3600
    Mt_Nm = Mt_exp_kgfm[i] * 9.80665

    print(f"\n=================================================")
    print(f"INICIANDO SIMULAÇÃO PARA O CASO: rv = {rv}")
    print(f"=================================================")

    # ==========================================================================
    # TAREFA: Felipe
    # Objetivo: Estruturar os dados de entrada para o solver, encapsulando os
    #           parâmetros (geométricos, operacionais, termodinâmicos) na
    #           tupla `pars`, e executar a simulação para obter as curvas de
    #           propriedades (P, V, T, m) para cada caso.
    # --------------------------------------------------------------------------
    pars = (
        'fired', B, S, L, rv, n,
        360.*(np.pi/180.), -150.*(np.pi/180.), # ThIVO, ThIVC
        150.*(np.pi/180), -360.*(np.pi/180),  # ThEVO, ThEVC
        ThSOC, ThEOC, aWF, mWF,
        pint, Tint, pexh, Texh_K, phi, fuel,
    )

    V, m, T, p = oc.ottoCycle(Th, pars)

    # ==========================================================================
    # TAREFA: Augusto
    # Objetivo: Calcular Potências e pressões médias
    #
    # ==========================================================================
    # ------------------- Trabalho indicado (robusto em θ) -------------------
    dVdTh = np.gradient(V, Th)                 # [m³/rad]
    Wi = np.trapezoid(p * dVdTh, Th)           # [J/ciclo]
    Wi = abs(Wi)                               # garante sinal físico

    # ------------------------- Potências (kW) ------------------------------
    Ni_kW = (Wi * (Zc * (n/x))) / 1000.0       # indicada (kW)
    Ne_kW = (Mt_Nm * omega) / 1000.0           # efetiva a partir do torque (kW)
    Na_kW = Ni_kW - Ne_kW                       # atrito (kW)
    Nt_kW = (mpF_kg_s * PCI_CH4) / 1000.0      # térmica (kW) com PCI em J/kg

    # ------------------ MEPs usando Vu (swept volume) ----------------------
    # imep = (Pot_indicada*1000)*x / (Vu * Zc * n)  [Pa] -> /1e3 [kPa]
    denom = (Vu * Zc * n)
    imep_kPa = ((Ni_kW * 1e3) * x / denom) / 1e3
    bmep_kPa = ((Ne_kW * 1e3) * x / denom) / 1e3
    fmep_kPa = (((Ni_kW - Ne_kW) * 1e3) * x / denom) / 1e3

    # ==========================================================================
    # TAREFA: Pedro
    # Objetivo: Calcular Rendimentos
    #
    # ==========================================================================

    # --------------------------Rendimentos -----------------------------------

    nt_pct = 100.0 * (Ni_kW / Nt_kW) if Nt_kW > 0 else np.nan   # térmico
    nm_pct = 100.0 * (Ne_kW / Ni_kW) if Ni_kW > 0 else np.nan   # mecânico
    ng_pct = 100.0 * (Ne_kW / Nt_kW) if Nt_kW > 0 else np.nan   # global
    mar_teo = rho_ar * (Vu + Vc) * (n/x) #massa de ar teórica para rendimento vol.
    mar_real = AFreal * mpF_kg_s # vazão mássica real de ar (kg/s)
    nv_pct = 100.0 * (mar_real/mar_teo)


    # ------------------ Consumo específico (g/kWh) -------------------------
    sfc_g_kWh = (mpF_kg_s / Ne_kW) * 3600.0 * 1e3 if Ne_kW > 0 else np.nan

    # Armazenar os resultados para uso posterior
    caso_atual = {
        'rv': rv,
        'V_sim': V,
        'm_sim': m,
        'T_sim': T,
        'p_sim': p,
        'Th_sim': Th,
        'Mt_exp': Mt_Nm,
        'mpF_exp': mpF_kg_s,
        'Ni': Ni_kW,
        'Ne': Ne_kW,
        'Na': Na_kW,
        'Nt': Nt_kW,
        'imep': imep_kPa,
        'bmep': bmep_kPa,
        'fmep': fmep_kPa,
        'nt': nt_pct,
        'nm': nm_pct,
        'ng': ng_pct,
        'nv': nv_pct
    }
    resultados_finais.append(caso_atual)
    print(f"--> Simulação para rv = {rv} concluída e resultados armazenados.")


# ==========================================================================
# TAREFA: Gustavo
# Objetivo: Gerar gráficos (ex: PxV) e tabelas. Salvar arquivos.
# --------------------------------------------------------------------------

# --- Volumes (m³) ---
V1 = resultados_finais[0]['V_sim']   # caso rv=9
V2 = resultados_finais[1]['V_sim']   # caso rv=10
V3 = resultados_finais[2]['V_sim']   # caso rv=11

# --- Pressões (kPa) ---
p1 = resultados_finais[0]['p_sim']
p2 = resultados_finais[1]['p_sim']
p3 = resultados_finais[2]['p_sim']

# --- Ângulo do virabrequim (em graus) ---
CAD = np.degrees(resultados_finais[0]['Th_sim'])

# Diagramas P-V
plt.figure(1)

plt.subplot(2, 2, 1)
plt.plot(V1, p1, color='r', linestyle='-')
plt.title("Diagrama p-V Ciclo Otto (rv = 9)")
plt.xlabel("Volume (m³)")
plt.ylabel("Pressão (Pa)")

plt.subplot(2, 2, 2)
plt.plot(V2, p2, color='b', linestyle='-')
plt.title("Diagrama p-V Ciclo Otto (rv = 10)")
plt.xlabel("Volume (m³)")
plt.ylabel("Pressão (Pa)")

plt.subplot(2, 2, 3)
plt.plot(V3, p3, color='g', linestyle='-')
plt.title("Diagrama p-V Ciclo Otto (rv = 11)")
plt.xlabel("Volume (m³)")
plt.ylabel("Pressão (Pa)")

plt.subplot(2, 2, 4)
plt.plot(V1, p1, label='rv = 9', color='r', linestyle='-')
plt.plot(V2, p2, label='rv = 10', color='b', linestyle='-')
plt.plot(V3, p3, label='rv = 11', color='g', linestyle='-')
plt.title("Diagrama p-V Ciclo Otto (diferentes valores de rv)")
plt.xlabel("Volume (m³)")
plt.ylabel("Pressão (Pa)")
plt.legend()

# Gráficos de Pressão x CAD
plt.figure(2)

plt.subplot(2, 2, 1)
plt.plot(CAD, p1, color='r', linestyle='-')
plt.title("Pressão pelo Ângulo do Virabrequim (rv = 9)")
plt.xlabel("Ângulo do Virabrequim (°)")
plt.ylabel("Pressão (Pa)")

plt.subplot(2, 2, 2)
plt.plot(CAD, p2, color='b', linestyle='-')
plt.title("Pressão pelo Ângulo do Virabrequim (rv = 10)")
plt.xlabel("Ângulo do Virabrequim (°)")
plt.ylabel("Pressão (Pa)")

plt.subplot(2, 2, 3)
plt.plot(CAD, p3, color='g', linestyle='-')
plt.title("Pressão pelo Ângulo do Virabrequim (rv = 11)")
plt.xlabel("Ângulo do Virabrequim (°)")
plt.ylabel("Pressão (Pa)")

plt.subplot(2, 2, 4)
plt.plot(CAD, p1, label='rv = 9', color='r', linestyle='-')
plt.plot(CAD, p2, label='rv = 10', color='b', linestyle='-')
plt.plot(CAD, p3, label='rv = 11', color='g', linestyle='-')
plt.title("Pressão pelo Ângulo do Virabrequim")
plt.xlabel("Ângulo do Virabrequim (°)")
plt.ylabel("Pressão (Pa)")
plt.legend()

# Gráficos de Volume x CAD
plt.figure(3)

plt.subplot(2, 2, 1)
plt.plot(CAD, V1, color='r', linestyle='-')
plt.title("Volume pelo Ângulo do Virabrequim (rv = 9)")
plt.xlabel("Ângulo do Virabrequim (°)")
plt.ylabel("Volume (m³)")

plt.subplot(2, 2, 2)
plt.plot(CAD, V2, color='b', linestyle='-')
plt.title("Volume pelo Ângulo do Virabrequim (rv = 10)")
plt.xlabel("Ângulo do Virabrequim (°)")
plt.ylabel("Volume (m³)")

plt.subplot(2, 2, 3)
plt.plot(CAD, V3, color='g', linestyle='-')
plt.title("Volume pelo Ângulo do Virabrequim (rv = 11)")
plt.xlabel("Ângulo do Virabrequim (°)")
plt.ylabel("Volume (m³)")

plt.subplot(2, 2, 4)
plt.plot(CAD, V1, label='rv = 9', color='r', linestyle='-')
plt.plot(CAD, V2, label='rv = 10', color='b', linestyle='-')
plt.plot(CAD, V3, label='rv = 11', color='g', linestyle='-')
plt.title("Volume pelo Ângulo do Virabrequim")
plt.xlabel("Ângulo do Virabrequim (°)")
plt.ylabel("Volume (m³)")
plt.legend()

# --- SALVAR FIGURAS DENTRO DO SCRIPT ---
import os, shutil
shutil.rmtree('figs', ignore_errors=True)
os.makedirs('figs', exist_ok=True)

# salva todas as figuras abertas neste processo
for i in plt.get_fignums():
    plt.figure(i)
    plt.tight_layout()
    plt.savefig(f'figs/figure_{i}.png', dpi=300, bbox_inches='tight')

# opcional: também em PDF/SVG
#     plt.savefig(f'figs/figure_{i}.pdf', bbox_inches='tight')
#     plt.savefig(f'figs/figure_{i}.svg', bbox_inches='tight')

# compacta em figs.zip
shutil.make_archive('figs', 'zip', 'figs')

# se quiser ainda ver inline quando rodar local/colab:

plt.show()

print("\n\n🏁 Simulações finalizadas.")

# ==========================================================================
# TAREFA: PESSOA 6
# Objetivo: Apresentar a tabela final compilada com os resultados.
# -------------------------------------------------------------------------
# ==========================================================================
print("\n==================== TABELA FINAL DE RESULTADOS ====================\n")

# Cabeçalho da tabela
print(f"{'rv':>3} | {'Ni (kW)':>9} | {'Ne (kW)':>9} | {'Na (kW)':>9} | {'Nt (kW)':>9} | "
      f"{'imep (kPa)':>11} | {'bmep (kPa)':>11} | {'fmep (kPa)':>11} | "
      f"{'ηt (%)':>7} | {'ηm (%)':>7} | {'ηg (%)':>7} | {'ηv (%)':>7} | {'Ce (kg/kJ)':>12}")

print("-" * 120)

# Linhas da tabela
for r in resultados_finais:
    Ce_kg_kJ = (r['mpF_exp'] / r['Ne']) if r['Ne'] > 0 else np.nan
    print(f"{r['rv']:>3.0f} | {r['Ni']:>9.3f} | {r['Ne']:>9.3f} | {r['Na']:>9.3f} | {r['Nt']:>9.3f} | "
          f"{r['imep']:>11.3f} | {r['bmep']:>11.3f} | {r['fmep']:>11.3f} | "
          f"{r['nt']:>7.2f} | {r['nm']:>7.2f} | {r['ng']:>7.2f} | {r['nv']:>7.2f} | {Ce_kg_kJ:>12.6f}")



#-----------------------------------------------------------------------------#
#=============================================================================#
#-----------------------------------------------------------------------------#
