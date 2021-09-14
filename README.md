# RST_LSTM

This is a Pytorch implementation of RST_LSTM, a recurrent model for radar echo extrapolation (precipitation nowcasting) as described in the following paper:

The Reconstitution Predictive Network for Precipitation Nowcasting, by Chuyao Luo, Guangning Xu, Xutao Li, Yunming Ye.

# Setup

Required python libraries: torch (>=1.7.0) + opencv + numpy + scipy (== 1.0.0) + jpype1.
Tested in ubuntu + nvidia 3090 Ti with cuda (>=11.0).

# Datasets
We conduct experiments on CIKM AnalytiCup 2017 datasets: [CIKM_AnalytiCup_Address](https://tianchi.aliyun.com/competition/entrance/231596/information) or [CIKM_Rardar](https://drive.google.com/drive/folders/1IqQyI8hTtsBbrZRRht3Es9eES_S4Qv2Y?usp=sharing) 

# Training
Use any '.py' script to train these models. To train the proposed model on the radar, we can simply run the cikm_rst_lstm_run.py

You might want to change the parameter and setting, you can change the details of variable ‘args’ in each files for each model

The preprocess method and data root path can be modified in the data/data_iterator.py file





