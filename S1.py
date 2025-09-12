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
    # Objetivo: Montar a tupla `pars` e chamar a função do ciclo Otto.
    # --------------------------------------------------------------------------
    pars = (
        'fired', B, S, L, rv, n,
        360.*(np.pi/180.), -150.*(np.pi/180.), # ThIVO, ThIVC
        150.*(np.pi/180), -360.*(np.pi/180),  # ThEVO, ThEVC
        ThSOC, ThEOC, aWF, mWF,
        pint, Tint, pexh, Texh_K, phi, fuel,
    )

    V, m, T, p = oc.ottoCycle(Th, pars)
    print("--> Simulação a ser executada aqui.")

    # ==========================================================================
    # TAREFA: PESSOA 3
    # Objetivo: Calcular Potências e Pressões Médias.
            # Geometria e volumes
    Vd = (np.pi/4.0) * (B**2) * S          # Volume deslocado (m³)
    cycles_per_sec = n / 2.0               # 4 tempos → 1 ciclo a cada 2 voltas

    # --- Preparação dos dados (garante arrays e finitos) ---
    p = np.asarray(p, float)
    V = np.asarray(V, float)
    T = np.asarray(T, float)
    m = np.asarray(m, float)

    good = np.isfinite(p) & np.isfinite(V) & np.isfinite(T) & np.isfinite(m)
    if not np.any(good):
        raise RuntimeError("Sem pontos válidos de p,V,T,m para integrar.")

    # --- dV/dθ e produto p*dV/dθ ---
    dVdTh = np.gradient(V, Th)             # m³/rad
    pdV    = p * dVdTh

    # --- Janela de VÁLVULAS FECHADAS: IVC -> EVO (gross work) ---
    ThIVO, ThIVC, ThEVO, ThEVC = pars[6], pars[7], pars[8], pars[9]
    mask_core = (Th > ThIVC) & (Th < ThEVO) & good

    if np.count_nonzero(mask_core) < 2:
        # fallback: integra no ciclo todo (pode incluir bombeamento)
        Wi_gross = np.trapezoid(pdV[good], Th[good])
    else:
        Wi_gross = np.trapezoid(pdV[mask_core], Th[mask_core])  # J/ciclo

    # orientação do laço (garante positivo)
    if Wi_gross < 0.0:
        Wi_gross = -Wi_gross

    # Considera IMEP como IMEPg (sem bombeamento)
    Wi = Wi_gross
    IMEPg = Wi_gross / Vd                   # Pa
    IMEP  = IMEPg                           # aqui usamos gross como IMEP (válvulas fechadas)
    IMEPpm = 0.0                            # não contamos bombeamento neste estágio

    # Potências
    Pi  = Wi * cycles_per_sec               # Potência indicada (W)
    Pe  = 2.0 * np.pi * n * Mt_Nm           # Potência efetiva (W) via torque
    Pth = mpF_kg_s * PCI_CH4                # Potência térmica (W)

    # BMEP (freio) pelo torque
    BMEP = (4.0 * np.pi * Mt_Nm) / Vd       # Pa

    # Rendimentos
    eta_i = Pi / Pth if Pth > 0 else np.nan
    eta_m = Pe / Pi  if Pi  > 0 else np.nan

    # Guarda no agregado
    resultados_finais.append({
        'rv': float(rv),
        'Texh_K': float(Texh_K),
        'm_dot_fuel_kg_s': float(mpF_kg_s),
        'Torque_Nm': float(Mt_Nm),
        'Wi_gross_J_per_cycle': float(Wi_gross),
        'Pi_W': float(Pi),
        'Pe_W': float(Pe),
        'Pth_W': float(Pth),
        'IMEPg_Pa': float(IMEPg),
        'IMEP_Pa': float(IMEP),
        'IMEPpump_Pa': float(IMEPpm),
        'BMEP_Pa': float(BMEP),
        'eta_i': float(eta_i),
        'eta_m': float(eta_m),
    })

    # Print resumido
    print(f"--> Wi(gross, IVC→EVO) = {Wi_gross:.2f} J/ciclo | "
          f"IMEPg = {IMEPg/1e5:.2f} bar | BMEP = {BMEP/1e5:.2f} bar")
    print(f"    Pi = {Pi/1e3:.2f} kW | Pe = {Pe/1e3:.2f} kW | Pth = {Pth/1e3:.2f} kW")
    print(f"    η_i = {eta_i*100:.1f}% | η_m = {eta_m*100:.1f}%")


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
