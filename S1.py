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

# Parâmetros da combustão (Função de Wiebe para S1)
ThSOC = -10. * (np.pi/180) # Início da combustão (rad)
ThEOC = +10. * (np.pi/180) # Fim da combustão (rad)
aWF = 5.0          # Fator de eficiência de Wiebe
mWF = 2.0          # Fator de forma de Wiebe

# Parâmetros de contorno (ambiente do laboratório)
pint = 100e3       # Pressão de admissão (Pa)
Tint = 273.15 + 25 # Temperatura de admissão (K)
pexh = 100e3       # Pressão de escape (Pa)
phi = 1.0          # Razão de equivalência

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

    # Armazenar os resultados para uso posterior
    caso_atual = {
        'rv': rv,
        'V_sim': V,
        'm_sim': m,
        'T_sim': T,
        'p_sim': p,
        'Th_sim': Th,
        'Mt_exp': Mt_Nm,
        'mpF_exp': mpF_kg_s
    }
    resultados_finais.append(caso_atual)
    print(f"--> Simulação para rv = {rv} concluída e resultados armazenados.")

    # Exemplo de como acessar os dados do primeiro caso (rv=9) após o loop:
    # print("\nDados armazenados para o primeiro caso (rv=9):")
    # print(resultados_finais[0]['p_sim']) # Imprime o array de pressão

    # ==========================================================================
    # TAREFA: PESSOA 3
    # Objetivo: Calcular Potências e Pressões Médias.

    Zc = 1                 # nº de cilindros (S1)
x  = 2                 # motor 4T -> 2 voltas por ciclo
Vu = (np.pi/4.0) * B**2 * S
omega = 2.0*np.pi*n    # rad/s

# (Re)inicializa contêiner com métricas finais por caso
metricas_finais = []

# Reexecuta/garante cálculos dentro do loop principal
resultados_finais = []
for i in range(len(rv_exp)):
    rv = rv_exp[i]
    Texh_K = Texh_exp_C[i] + 273.15
    mpF_kg_s = mpF_exp_kg_h[i] / 3600.0
    Mt_Nm = Mt_exp_kgfm[i] * 9.80665

    print(f"\n=================================================")
    print(f"INICIANDO SIMULAÇÃO PARA O CASO: rv = {rv}")
    print(f"=================================================")

    pars = (
        'fired', B, S, L, rv, n,
        360.*(np.pi/180.), -150.*(np.pi/180.),  # ThIVO, ThIVC
        150.*(np.pi/180.), -360.*(np.pi/180.),  # ThEVO, ThEVC
        ThSOC, ThEOC, aWF, mWF,
        pint, Tint, pexh, Texh_K, phi, fuel,
    )

    V, m, T, p = oc.ottoCycle(Th, pars)

    # ------------------- Trabalho indicado (robusto em θ) -------------------
    dVdTh = np.gradient(V, Th)                 # [m³/rad]
    Wi = np.trapezoid(p * dVdTh, Th)           # [J/ciclo]
    Wi = abs(Wi)                               # garante sinal físico

    # ------------------------- Potências (kW) ------------------------------
    Ni_kW = (Wi * (Zc * (n/x))) / 1000.0       # indicada (kW)
    Ne_kW = (Mt_Nm * omega) / 1000.0           # efetiva a partir do torque (kW)
    Na_kW = Ni_kW - Ne_kW                       # atrito (kW)

    # ---------------------- Potência térmica e rendimentos ------------------
    Nt_kW = (mpF_kg_s * PCI_CH4) / 1000.0      # térmica (kW) com PCI em J/kg
    nt_pct = 100.0 * (Ni_kW / Nt_kW) if Nt_kW > 0 else np.nan   # térmico
    nm_pct = 100.0 * (Ne_kW / Ni_kW) if Ni_kW > 0 else np.nan   # mecânico
    ng_pct = 100.0 * (Ne_kW / Nt_kW) if Nt_kW > 0 else np.nan   # global

    # ------------------ MEPs usando Vu (swept volume) ----------------------
    # imep = (Pot_indicada*1000)*x / (Vu * Zc * n)  [Pa] -> /1e3 [kPa]
    denom = (Vu * Zc * n)
    imep_kPa = ((Ni_kW * 1e3) * x / denom) / 1e3
    bmep_kPa = ((Ne_kW * 1e3) * x / denom) / 1e3
    fmep_kPa = (((Ni_kW - Ne_kW) * 1e3) * x / denom) / 1e3

    # ------------------ Consumo específico (g/kWh) -------------------------
    sfc_g_kWh = (mpF_kg_s / Ne_kW) * 3600.0 * 1e3 if Ne_kW > 0 else np.nan

    # Armazena tudo do caso
    caso_atual = {
        'rv': rv,
        'Th_sim': Th, 'V_sim': V, 'p_sim': p, 'T_sim': T, 'm_sim': m,
        'Mt_exp': Mt_Nm, 'mpF_exp': mpF_kg_s,
        'Wi_J_ciclo': Wi,
        'Ni_kW': Ni_kW, 'Ne_kW': Ne_kW, 'Na_kW': Na_kW, 'Nt_kW': Nt_kW,
        'eta_term_pct': nt_pct, 'eta_mec_pct': nm_pct, 'eta_glob_pct': ng_pct,
        'imep_kPa': imep_kPa, 'bmep_kPa': bmep_kPa, 'fmep_kPa': fmep_kPa,
        'sfc_g_kWh': sfc_g_kWh
    }
    resultados_finais.append(caso_atual)

    # Print-resumo do caso
    print(f"--> rv={rv:.1f} | Ni={Ni_kW:.2f} kW | Ne={Ne_kW:.2f} kW | Nt={Nt_kW:.2f} kW")
    print(f"    MEPs: imep={imep_kPa:.1f} kPa | bmep={bmep_kPa:.1f} kPa | fmep={fmep_kPa:.1f} kPa")
    print(f"    ηt={nt_pct:.1f}% | ηm={nm_pct:.1f}% | ηg={ng_pct:.1f}% | SFC={sfc_g_kWh:.1f} g/kWh")

# --------------------------------------------------------------------------
# 6. Tabela-resumo final (linhas por caso)
# --------------------------------------------------------------------------
print("\n========== RESUMO FINAL ==========")
print("rv |  Ni[kW]  Ne[kW]  Nt[kW] | imep[kPa] bmep[kPa] fmep[kPa] | ηt[%] ηm[%] ηg[%] | SFC[g/kWh]")
for r in resultados_finais:
    print(f"{r['rv']:>2.0f} | {r['Ni_kW']:>7.2f} {r['Ne_kW']:>7.2f} {r['Nt_kW']:>7.2f} | "
          f"{r['imep_kPa']:>8.1f} {r['bmep_kPa']:>9.1f} {r['fmep_kPa']:>9.1f} | "
          f"{r['eta_term_pct']:>5.1f} {r['eta_mec_pct']:>5.1f} {r['eta_glob_pct']:>5.1f} | "
          f"{r['sfc_g_kWh']:>9.1f}")

    # ==========================================================================
    # TAREFA: PESSOA 4
    # Objetivo: Calcular os Rendimentos.
    # --------------------------------------------------------------------------
    print("--> Cálculos de Rendimento a serem implementados aqui.")

    # ==========================================================================
    # TAREFA: PESSOA 5
    # Objetivo: Gerar gráficos (ex: PxV) e tabelas. Salvar arquivos.
    # --------------------------------------------------------------------------
    print("--> Geração de gráficos e tabelas a ser implementada aqui.")

print("\n\n🏁 Simulações finalizadas.")

# ==========================================================================
# TAREFA: PESSOA 6
# Objetivo: Apresentar a tabela final compilada com os resultados.
# -------------------------------------------------------------------------
print("\n--> Tabela final de resultados a ser gerdX.")


#-----------------------------------------------------------------------------#
#=============================================================================#
#-----------------------------------------------------------------------------#
