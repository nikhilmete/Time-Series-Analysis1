# Time Series Analysis with RNN, LSTM & GRU

A complete, hands-on tutorial on **Time Series Forecasting** using classical methods and deep learning (RNN → LSTM → GRU → Bidirectional LSTM).

## What You Will Learn

- Time series fundamentals: trend, seasonality, stationarity, ACF & PACF
- Classical decomposition and a simple moving-average baseline
- Intuitive theory of why RNNs struggle with long-term dependencies
- How **LSTM** and **GRU** solve the vanishing gradient problem
- Building and training:
  - LSTM
  - GRU
  - Bidirectional LSTM (BiLSTM)
- Model comparison using RMSE and MAE
- Recursive multi-step forecasting of future days (genuine unseen forecast)

## Project Structure (suggested)

```
time-series-rnn-lstm-gru/
├── README.md
├── requirements.txt
├── notebooks/
│   └── Time_Series_Analysis_RNN_LSTM_GRU_Tutorial.ipynb
├── data/                     # optional: your own CSV
└── src/                      # optional: extracted helper functions
```

## Requirements

- Python 3.9+
- TensorFlow 2.x

```bash
pip install -r requirements.txt
```

## Quick Start

1. Open the notebook:
   ```bash
   jupyter notebook notebooks/Time_Series_Analysis_RNN_LSTM_GRU_Tutorial.ipynb
   ```
2. Run all cells.
3. The notebook will:
   - Load / generate the sales time series
   - Perform EDA and classical decomposition
   - Create sliding windows
   - Train LSTM, GRU and BiLSTM models
   - Compare them against a moving-average baseline
   - Produce a 14-day future forecast

## Key Concepts Covered

| Topic                    | Description                                      |
|--------------------------|--------------------------------------------------|
| Stationarity             | Why it matters and how to check it               |
| ACF / PACF               | Identifying lag structure                        |
| Sliding Window           | Converting time series into supervised learning  |
| LSTM                     | Long Short-Term Memory cells                     |
| GRU                      | Gated Recurrent Unit (simpler alternative)       |
| BiLSTM                   | Looking at past and future context               |
| Recursive Forecasting    | Multi-step prediction by feeding predictions back|

## Model Comparison

The notebook trains multiple models and ranks them by **RMSE** and **MAE** on the test set.  
A classical moving-average baseline is also included so you can see when deep learning is (and isn’t) worth the extra complexity.

## Future Forecast

The final section uses the best model to forecast **14 genuinely unseen future days** by recursively feeding each prediction back into the input window.

## Notes

- Deep learning does not always beat simple baselines on smooth, strongly seasonal data.  
  This is an important real-world lesson the notebook deliberately highlights.
- You can easily change the `WINDOW` size, add extra features (day-of-week, holidays, etc.), or stack more recurrent layers.

## License

Free for learning and personal projects.
