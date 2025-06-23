from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Carregando modelo e vetorizador
rfc = joblib.load('../modelos/rfc.pkl')
dtc = joblib.load('../modelos/dtc.pkl')
lr = joblib.load('../modelos/lr.pkl')
lsvc = joblib.load('../modelos/lsvc.pkl')
pac = joblib.load('../modelos/pac.pkl')
vetorizador_n = joblib.load('../modelos/vetorizador_juncao_n.pkl')
vetorizador_c = joblib.load('../modelos/vetorizador_juncao_c.pkl')

@app.route('/prever', methods=['POST'])
def prever():
    dados = request.get_json()
    texto = dados['texto']

    vetor_n = vetorizador_n.transform([texto])
    vetor_c = vetorizador_c.transform([texto])
    pred_rfc = rfc.predict(vetor_c)
    pred_dtc = dtc.predict(vetor_c)
    pred_lr = lr.predict(vetor_n)
    pred_lsvc = lsvc.predict(vetor_c)
    pred_pac = pac.predict(vetor_c)

    return jsonify({'rfc': int(pred_rfc[0]),
                    'dtc': int(pred_dtc[0]),
                    'logreg': int(pred_lr[0]),
                    'svc': int(pred_lsvc[0]),
                    'pac': int(pred_pac[0])})

@app.route('/')
def root():
    return 'API ativa'

if __name__ == '__main__':
    app.run(debug=True)