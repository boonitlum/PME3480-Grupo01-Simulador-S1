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
    # TAREFA: FELIPE
    # Objetivo: Montar a tupla `pars` e chamar a função do ciclo Otto.
    # --------------------------------------------------------------------------
    pars = (
        'fired', B, S, L, rv, n,
        360.*(np.pi/180.), -150.*(np.pi/180.), # IVO, IVC
        150.*(np.pi/180), -360.*(np.pi/180),  # EVO, EVC
        ThSOC, ThEOC, aWF, mWF,
        pint, Tint, pexh, Texh_K, phi, fuel,
    )

    # Descomente a linha abaixo para rodar a simulação
    # V, m, T, p = oc.ottoCycle(Th, pars)
    print("--> Simulação a ser executada aqui.")

    # ==========================================================================
    # TAREFA: PESSOA 3
    # Objetivo: Calcular Potências e Pressões Médias.
    # --------------------------------------------------------------------------
    print("--> Cálculos de Potência e Pressão a serem implementados aqui.")

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
# --------------------------------------------------------------------------
print("\n--> Tabela final de resultados a ser gerada aqui.")


#-----------------------------------------------------------------------------#
#=============================================================================#
#-----------------------------------------------------------------------------#
