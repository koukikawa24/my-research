使い方

1. 音声をその場で録音する場合
estimate_age.pyを実行する。

2. wavファイルを読み込む場合
estimate_age.pyの151行目の audio = record_audio(duration=4.0, fs=16000)をコメントアウトし、
関数load_audio_from_wavを使用して読み込むwavファイルのパスを指定する。
なおwavファイルは、サンプリングレート16kHz, 4秒以上のものを使用すること。