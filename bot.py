from tokenize import group
import pandas as pd
import yfinance as yf
from backtesting import Strategy
from backtesting.lib import FractionalBacktest

# import pandas_ta as ta  # A função fractal será implementada manualmente

# --- DEFINIÇÃO DA ESTRATÉGIA COM FILTRO DE FRACTAL ---


class FractalReversalStrategy(Strategy):
    # Parâmetros da estratégia
    lookback_snr = 50
    lookback_corpo = 20
    fator_candle_forca = 1.8
    fator_max_pavio = 0.25
    min_risco_retorno = 1.5

    def init(self):
        # A lógica de preparação dos indicadores foi movida para fora da classe
        pass

    def next(self):
        # Função executada para cada candle
        if self.position:
            return

        # --- LÓGICA DE COMPRA (REVERSÃO EM SUPORTE) ---
        # Verificamos se um fractal de alta foi CONFIRMADO na vela anterior.
        # Lembre-se do lag: a confirmação em [-1] refere-se ao fundo em [-3].
        if self.data.BullishFractal[-1]:
            preco_fundo_fractal = self.data.Low[-3]
            suporte_atual = self.data.SupZoneLow[-3]

            # CONFLUÊNCIA 1: O fundo do fractal está na zona de suporte?
            if preco_fundo_fractal <= suporte_atual * 1.005:
                # CONFLUÊNCIA 2: O candle ATUAL é um gatilho de compra predominante?
                if self.data.IsBullishPredominant[-1]:
                    # Lógica de Risco/Retorno e execução
                    resistencia_alvo = self.data.ResZoneHigh[-1]
                    stop_loss = preco_fundo_fractal * 0.998
                    retorno = resistencia_alvo - self.data.Close[-1]
                    risco = self.data.Close[-1] - stop_loss

                    if risco > 0 and retorno / risco >= self.min_risco_retorno:
                        self.buy(size=0.95, sl=stop_loss, tp=resistencia_alvo)

        # --- LÓGICA DE VENDA (REVERSÃO EM RESISTÊNCIA) ---
        # Verificamos se um fractal de baixa foi CONFIRMADO na vela anterior.
        # A confirmação em [-1] refere-se ao topo em [-3].
        if self.data.BearishFractal[-1]:
            preco_topo_fractal = self.data.High[-3]
            resistencia_atual = self.data.ResZoneHigh[-3]

            # CONFLUÊNCIA 1: O topo do fractal está na zona de resistência?
            if preco_topo_fractal >= resistencia_atual * 0.995:
                # CONFLUÊNCIA 2: O candle ATUAL é um gatilho de venda predominante?
                if self.data.IsBearishPredominant[-1]:
                    # Lógica de Risco/Retorno e execução
                    suporte_alvo = self.data.SupZoneLow[-1]
                    stop_loss = preco_topo_fractal * 1.002
                    retorno = self.data.Close[-1] - suporte_alvo
                    risco = stop_loss - self.data.Close[-1]

                    if risco > 0 and retorno / risco >= self.min_risco_retorno:
                        self.sell(size=0.95, sl=stop_loss, tp=suporte_alvo)


# --- PREPARAÇÃO DOS DADOS E INDICADORES ---
df = yf.download("BTC-USD", period="1y", interval="4h")

# Garante que o índice de colunas seja de nível único
df.columns = df.columns.get_level_values(0)

# --- CÁLCULO DOS INDICADORES USANDO PANDAS ---
lookback_snr = 50
lookback_corpo = 20
fator_candle_forca = 1.8
fator_max_pavio = 0.25

df["body_size"] = abs(df["Close"] - df["Open"])
df["upper_wick"] = df["High"] - df[["Open", "Close"]].max(axis=1)
df["lower_wick"] = df[["Open", "Close"]].min(axis=1) - df["Low"]

df["ResZoneHigh"] = df["High"].rolling(lookback_snr).max()
df["SupZoneLow"] = df["Low"].rolling(lookback_snr).min()

media_corpo = df["body_size"].rolling(lookback_corpo).mean()
is_force_candle = df["body_size"] > media_corpo * fator_candle_forca

df["IsBullishPredominant"] = (
    (df["Close"] > df["Open"])
    & (df["upper_wick"] < df["body_size"] * fator_max_pavio)
    & is_force_candle
)

df["IsBearishPredominant"] = (
    (df["Close"] < df["Open"])
    & (df["lower_wick"] < df["body_size"] * fator_max_pavio)
    & is_force_candle
)

# --- CÁLCULO MANUAL DOS FRACTAIS ---
# Um fractal de baixa (topo) é um High maior que os 2 Highs anteriores e os 2 posteriores.
df["BearishFractal"] = (
    (df["High"] > df["High"].shift(1))
    & (df["High"] > df["High"].shift(2))
    & (df["High"] > df["High"].shift(-1))
    & (df["High"] > df["High"].shift(-2))
)

# Um fractal de alta (fundo) é um Low menor que os 2 Lows anteriores e os 2 posteriores.
df["BullishFractal"] = (
    (df["Low"] < df["Low"].shift(1))
    & (df["Low"] < df["Low"].shift(2))
    & (df["Low"] < df["Low"].shift(-1))
    & (df["Low"] < df["Low"].shift(-2))
)


# --- EXECUÇÃO DO BACKTEST ---
bt = FractionalBacktest(df, FractalReversalStrategy, cash=100000, commission=0.001)

stats = bt.run()
print(stats)
bt.plot()
