

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================
st.set_page_config(
    page_title="Painel Quantitativo Esportivo",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 1. CORE MATH ENGINE (CLASSES MESTRE)
# =============================================================================
class ModeloMestreEsportivo:
    def __init__(self, banca_total=1000.0, ev_minimo=3.0, fracao_kelly=0.25):
        self.banca_total = banca_total
        self.ev_minimo = ev_minimo
        self.fracao_kelly = fracao_kelly

    def _poisson_pmf(self, k, lambd):
        return (lambd**k * math.exp(-lambd)) / math.factorial(k)

    def _avaliar_ev_e_kelly(self, prob_modelo, odd_mercado):
        p = prob_modelo / 100.0
        b = odd_mercado - 1.0
        q = 1.0 - p
        
        ev = ((p * b) - q) * 100.0
        
        if ev >= self.ev_minimo:
            f_kelly = ((p * b) - q) / b
            f_kelly_ajustado = max(0.0, f_kelly * self.fracao_kelly)
            stake = self.banca_total * f_kelly_ajustado
        else:
            f_kelly_ajustado = 0.0
            stake = 0.0
            
        return round(ev, 2), round(f_kelly_ajustado * 100, 2), round(stake, 2)

    def _modelo_futebol_poisson(self, xg_casa, xg_fora, max_gols=6):
        prob_under25 = 0.0
        for g_casa in range(max_gols + 1):
            for g_fora in range(max_gols + 1):
                if (g_casa + g_fora) < 2.5:
                    p = self._poisson_pmf(g_casa, xg_casa) * self._poisson_pmf(g_fora, xg_fora)
                    prob_under25 += p
        
        prob_over25 = (1.0 - prob_under25) * 100.0
        return {"Mercado": "Over 2.5 Gols", "Probabilidade (%)": round(prob_over25, 2)}

    def _modelo_basquete_gaussiano(self, media_casa, desvio_casa, media_fora, desvio_fora, linha_handicap_casa):
        media_diff = media_casa - media_fora
        desvio_diff = math.sqrt(desvio_casa**2 + desvio_fora**2)
        prob_handicap = (1.0 - stats.norm.cdf(linha_handicap_casa, loc=media_diff, scale=desvio_diff)) * 100.0
        return {"Mercado": f"Handicap Mandante ({linha_handicap_casa})", "Probabilidade (%)": round(prob_handicap, 2)}

    def _modelo_tenis_markov(self, p_saque_casa):
        p = p_saque_casa
        p_deuce = 20.0 * (p**3) * ((1.0 - p)**3)
        p_direto = (p**4) * (15.0 - 24.0*p + 10.0*(p**2))
        p_win_deuce = (p**2) / ((p**2) + ((1.0 - p)**2))
        prob_game = (p_direto + (p_deuce * p_win_deuce)) * 100.0
        return {"Mercado": "Confirmar Game de Saque", "Probabilidade (%)": round(prob_game, 2)}

    def _modelo_esports_bo3(self, prob_mapa):
        prob_2x0 = prob_mapa ** 2
        prob_2x1 = 2.0 * (prob_mapa ** 2) * (1.0 - prob_mapa)
        prob_bo3 = (prob_2x0 + prob_2x1) * 100.0
        return {"Mercado": "Vencer Série Bo3", "Probabilidade (%)": round(prob_bo3, 2)}

    def _modelo_corrida_plackett_luce(self, elos_pilotos, n_sims=5000):
        n_pilotos = len(elos_pilotos)
        vitorias = np.zeros(n_pilotos)
        for _ in range(n_sims):
            desempenho = elos_pilotos + np.random.gumbel(0, 100, n_pilotos)
            vitorias[np.argmax(desempenho)] += 1
        prob_vitoria_lider = (vitorias[0] / n_sims) * 100.0
        return {"Mercado": "Vitória Piloto Principal", "Probabilidade (%)": round(prob_vitoria_lider, 2)}

    def analisar(self, esporte, dados, odd_mercado=None):
        esporte_clean = esporte.lower().strip()
        
        if esporte_clean in ['futebol', 'nhl', 'hóquei', 'hoquei', 'futsal']:
            res = self._modelo_futebol_poisson(dados['xg_casa'], dados['xg_fora'])
        elif esporte_clean in ['basquete', 'nba', 'nbb']:
            res = self._modelo_basquete_gaussiano(dados['media_casa'], dados['desvio_casa'], dados['media_fora'], dados['desvio_fora'], dados['linha_handicap'])
        elif esporte_clean in ['tênis', 'tenis', 'vôlei', 'volei']:
            res = self._modelo_tenis_markov(dados['p_saque'])
        elif esporte_clean in ['esports', 'cs2', 'lol', 'valorant']:
            res = self._modelo_esports_bo3(dados['p_mapa'])
        elif esporte_clean in ['corrida', 'f1', 'motogp', 'nascar']:
            res = self._modelo_corrida_plackett_luce(dados['elos'])
        else:
            raise ValueError(f"Esporte '{esporte}' não mapeado.")

        if odd_mercado:
            ev, kelly, stake = self._avaliar_ev_e_kelly(res["Probabilidade (%)"], odd_mercado)
            res.update({
                "Odd Mercado": odd_mercado,
                "EV (%)": ev,
                "Kelly Ajustado (%)": kelly,
                "Stake Recomendada ($)": stake
            })
            
        return res


class BacktesterEsportivo:
    def __init__(self, banca_inicial=1000.0, fracao_kelly=0.25, ev_minimo=3.0):
        self.banca_inicial = banca_inicial
        self.fracao_kelly = fracao_kelly
        self.ev_minimo = ev_minimo

    def calcular_brier_score(self, probabilidades, resultados_reais):
        return np.mean((np.array(probabilidades) - np.array(resultados_reais)) ** 2)

    def executar(self, df_historico):
        df = df_historico.copy()
        df['p_modelo'] = df['prob_modelo'] / 100.0
        df['odd_b'] = df['odd_mercado'] - 1.0
        df['ev_pct'] = ((df['p_modelo'] * df['odd_b']) - (1.0 - df['p_modelo'])) * 100.0
        
        df['apostar'] = df['ev_pct'] >= self.ev_minimo
        f_kelly_puro = ((df['p_modelo'] * df['odd_b']) - (1.0 - df['p_modelo'])) / df['odd_b']
        df['stake_pct'] = np.where(df['apostar'], np.maximum(0, f_kelly_puro * self.fracao_kelly), 0.0)
        
        banca_atual = self.banca_inicial
        historico_banca = [banca_atual]
        lucros = []
        stakes_val = []
        
        for _, row in df.iterrows():
            if row['apostar']:
                stake = banca_atual * row['stake_pct']
                lucro = (stake * row['odd_b']) if row['resultado_real'] == 1 else -stake
                banca_atual += lucro
                lucros.append(lucro)
                stakes_val.append(stake)
            else:
                lucros.append(0.0)
                stakes_val.append(0.0)
            historico_banca.append(banca_atual)
            
        df['stake_dinheiro'] = stakes_val
        df['lucro_dinheiro'] = lucros
        df['banca_acumulada'] = historico_banca[1:]
        
        apostas_feitas = df[df['apostar']]
        total_apostas = len(apostas_feitas)
        
        if total_apostas > 0:
            brier_score = self.calcular_brier_score(df['p_modelo'], df['resultado_real'])
            brier_apostas = self.calcular_brier_score(apostas_feitas['p_modelo'], apostas_feitas['resultado_real'])
            brier_mercado = self.calcular_brier_score(1.0 / df['odd_mercado'], df['resultado_real'])
            
            total_investido = apostas_feitas['stake_dinheiro'].sum()
            lucro_total = banca_atual - self.banca_inicial
            roi_yield = (lucro_total / total_investido * 100) if total_investido > 0 else 0.0
            
            curva_banca = np.array(historico_banca)
            picos = np.maximum.accumulate(curva_banca)
            drawdowns = (curva_banca - picos) / picos * 100.0
            max_drawdown = np.min(drawdowns)
            taxa_acerto = (apostas_feitas['resultado_real'].sum() / total_apostas) * 100
        else:
            brier_score, brier_apostas, brier_mercado = 0, 0, 0
            total_investido, lucro_total, roi_yield, max_drawdown, taxa_acerto = 0, 0, 0, 0, 0

        relatorio = {
            "brier_modelo": round(brier_score, 4),
            "brier_apostas": round(brier_apostas, 4),
            "brier_mercado": round(brier_mercado, 4),
            "ganho_precisao": round(((brier_mercado - brier_score) / brier_mercado) * 100, 2) if brier_mercado > 0 else 0,
            "total_apostas": total_apostas,
            "taxa_acerto": round(taxa_acerto, 2),
            "banca_final": round(banca_atual, 2),
            "lucro_total": round(lucro_total, 2),
            "volume_investido": round(total_investido, 2),
            "yield_roi": round(roi_yield, 2),
            "max_drawdown": round(max_drawdown, 2)
        }
        
        return relatorio, df

# =============================================================================
# 2. INTERFACE SIDEBAR (CONFIGURAÇÕES GERAIS)
# =============================================================================
st.sidebar.title("⚙️ Gestão de Risco")
banca_input = st.sidebar.number_input("Banca Total ($)", value=1000.0, step=100.0)
ev_minimo_input = st.sidebar.slider("EV Mínimo Exigido (%)", min_value=0.0, max_value=15.0, value=3.0, step=0.5)
fracao_kelly_input = st.sidebar.select_slider(
    "Fração do Critério de Kelly",
    options=[0.10, 0.25, 0.50, 1.00],
    value=0.25,
    format_func=lambda x: f"{int(x*100)}% ({'Conservador' if x==0.25 else 'Agressivo' if x==1.0 else 'Moderado'})"
)

# Instancia o motor com os parâmetros do usuário
motor = ModeloMestreEsportivo(banca_total=banca_input, ev_minimo=ev_minimo_input, fracao_kelly=fracao_kelly_input)

# =============================================================================
# 3. CONSTRUÇÃO DO DASHBOARD (ABAS)
# =============================================================================
st.title("📊 Plataforma Quantitativa de Análise Esportiva")

aba1, aba2 = st.tabs(["🎯 Calculadora de Valor (+EV)", "📈 Backtesting Histórico & Brier Score"])

# -----------------------------------------------------------------------------
# ABA 1: CALCULADORA DE VALOR (+EV)
# -----------------------------------------------------------------------------
with aba1:
    st.subheader("Análise & Precificação em Tempo Real")
    col_esporte, col_odd = st.columns([2, 1])
    
    with col_esporte:
        esporte_sel = st.selectbox(
            "Selecione a Modalidade",
            ["Futebol", "Basquete (NBA)", "Tênis", "eSports (Bo3)", "Corrida (F1)"]
        )
    with col_odd:
        odd_mercado = st.number_input("Odd Atual do Mercado", value=1.95, min_value=1.01, step=0.05)

    st.markdown("---")
    st.markdown("##### 📥 Parâmetros de Entrada da Partida")

    dados_entrada = {}
    
    if esporte_sel == "Futebol":
        c1, c2 = st.columns(2)
        dados_entrada['xg_casa'] = c1.number_input("xG Esperado Mandante", value=2.10, step=0.10)
        dados_entrada['xg_fora'] = c2.number_input("xG Esperado Visitante", value=1.20, step=0.10)
        
    elif esporte_sel == "Basquete (NBA)":
        c1, c2, c3 = st.columns(3)
        dados_entrada['media_casa'] = c1.number_input("Média Pontos Mandante", value=118.0)
        dados_entrada['desvio_casa'] = c1.number_input("Desvio Padrão Mandante", value=8.0)
        dados_entrada['media_fora'] = c2.number_input("Média Pontos Visitante", value=112.0)
        dados_entrada['desvio_fora'] = c2.number_input("Desvio Padrão Visitante", value=9.0)
        dados_entrada['linha_handicap'] = c3.number_input("Linha Handicap Mandante", value=4.5)
        
    elif esporte_sel == "Tênis":
        dados_entrada['p_saque'] = st.slider("Probabilidade de Aritmética no Saque (Mandante)", 0.50, 0.95, 0.68)
        
    elif esporte_sel == "eSports (Bo3)":
        dados_entrada['p_mapa'] = st.slider("Probabilidade de Vencer 1 Mapa", 0.10, 0.90, 0.60)
        
    elif esporte_sel == "Corrida (F1)":
        st.caption("Insira os ELOS dos 3 principais pilotos para simulação de Monte Carlo:")
        e1 = st.number_input("Elo Piloto Favorito", value=1600)
        e2 = st.number_input("Elo Piloto 2", value=1500)
        e3 = st.number_input("Elo Piloto 3", value=1400)
        dados_entrada['elos'] = np.array([e1, e2, e3])

    if st.button("🚀 Executar Modelo Estatístico", use_container_width=True):
        mapeamento_esportes = {
            "Futebol": "futebol",
            "Basquete (NBA)": "nba",
            "Tênis": "tenis",
            "eSports (Bo3)": "esports",
            "Corrida (F1)": "f1"
        }
        
        resultado = motor.analisar(mapeamento_esportes[esporte_sel], dados_entrada, odd_mercado)
        
        st.markdown("### 📋 Resultado da Inspeção")
        m1, m2, m3, m4 = st.columns(4)
        
        m1.metric("Probabilidade Estimada", f"{resultado['Probabilidade (%)']}%")
        
        ev_val = resultado.get('EV (%)', 0)
        m2.metric("Valor Esperado (EV)", f"{ev_val}%", delta=f"{ev_val}%", delta_color="normal" if ev_val > 0 else "inverse")
        
        m3.metric("Kelly Ajustado", f"{resultado.get('Kelly Ajustado (%)', 0)}%")
        m4.metric("Stake Recomendada", f"${resultado.get('Stake Recomendada ($)', 0)}")
        
        if ev_val >= ev_minimo_input:
            st.success(f"✅ **Aposta Com Valor (+EV)!** Recomendação: Entrar no mercado `{resultado['Mercado']}` com stake de **${resultado['Stake Recomendada ($)']}**.")
        else:
            st.warning(f"⚠️ **Sem Valor Suficiente.** O EV atual ({ev_val}%) é inferior ao limiar exigido na barra lateral ({ev_minimo_input}%).")

# -----------------------------------------------------------------------------
# ABA 2: BACKTESTING HISTÓRICO & BRIER SCORE
# -----------------------------------------------------------------------------
with aba2:
    st.subheader("Simulação em Massa & Validação Probabilística")
    
    n_sims = st.number_input("Quantidade de Jogos Simulados", value=1000, step=100)
    
    if st.button("🔄 Rodar Backtest Histórico", use_container_width=True):
        np.random.seed(42)
        p_verdadeira = np.random.uniform(0.40, 0.70, n_sims)
        resultados_sim = np.random.binomial(1, p_verdadeira)
        odds_sim = np.round((1 / p_verdadeira) * 0.95, 2)
        p_modelo_sim = np.clip(p_verdadeira + np.random.normal(0, 0.04, n_sims), 0.1, 0.9) * 100

        df_sim = pd.DataFrame({
            'prob_modelo': p_modelo_sim,
            'odd_mercado': odds_sim,
            'resultado_real': resultados_sim
        })

        backtester = BacktesterEsportivo(banca_inicial=banca_input, fracao_kelly=fracao_kelly_input, ev_minimo=ev_minimo_input)
        relatorio, df_res = backtester.executar(df_sim)

        st.markdown("##### 📌 Métricas de Calibração & Brier Score")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Brier Score Modelo", relatorio['brier_modelo'], help="Quanto menor, melhor calibrado.")
        b2.metric("Brier Score Mercado", relatorio['brier_mercado'])
        b3.metric("Ganho vs Mercado", f"{relatorio['ganho_precisao']}%")
        b4.metric("Taxa de Acerto", f"{relatorio['taxa_acerto']}%")

        st.markdown("##### 💰 Performance Financeira")
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Lucro Líquido Total", f"${relatorio['lucro_total']}")
        f2.metric("Banca Final", f"${relatorio['banca_final']}")
        f3.metric("Yield / ROI", f"{relatorio['yield_roi']}%")
        f4.metric("Max Drawdown", f"{relatorio['max_drawdown']}%")

        st.markdown("##### 📈 Curva de Evolução Patrimonial")
        st.line_chart(df_res['banca_acumulada'])

pandas
numpy
scipy


