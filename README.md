# Plataforma de Análise de Imagens Médicas (Prototipo de Pesquisa)

> **Aviso importante:** este projeto é apenas um protótipo de pesquisa e não deve ser usado para emitir diagnósticos clínicos ou substituir a avaliação de profissionais de saúde. Os modelos produzidos com este código precisam passar por validações clínicas, revisão ética e aprovação regulatória antes de qualquer uso real.

## Visão geral

Este repositório fornece uma estrutura em Python para treinar e avaliar modelos de classificação de imagens médicas (por exemplo, ressonâncias magnéticas) utilizando `PyTorch`. O objetivo é detectar padrões associados a lesões comparando exames de pacientes com imagens de referência consideradas "normais".

O projeto inclui:

- scripts de treinamento, inferência e comparação de exames;
- pipeline de dados baseado em diretórios rotulados (`normal`, `lesao`, etc.);
- configuração por arquivos YAML;
- funções utilitárias para registro e checkpoints;
- serviço FastAPI com health-check e logs estruturados para monitoramento em produção.

## Estrutura de diretórios

```
.
├── configs/
│   └── default.yaml
├── data/
│   ├── train/
│   │   ├── normal/
│   │   └── lesao/
│   └── val/
│       ├── normal/
│       └── lesao/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── scripts/
│   ├── compare.py
│   ├── deploy.sh
│   ├── predict.py
│   └── train.py
└── src/
    └── medimaging_ai/
        ├── __init__.py
        ├── compare.py
        ├── api/
        │   ├── __init__.py
        │   └── app.py
        ├── config.py
        ├── data.py
        ├── inference.py
        ├── settings.py
        ├── models.py
        ├── trainer.py
        └── utils.py
```

## Instalação

1. Crie e ative um ambiente virtual (recomendado):

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows PowerShell
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Deploy com Docker

O repositório inclui uma infraestrutura pronta para containerização e publicação atrás de um proxy HTTPS (NGINX). Para iniciar rapidamente:

1. Copie o arquivo de exemplo e ajuste as variáveis de ambiente necessárias (tokens do WhatsApp, caminhos de checkpoints e parâmetros do modelo):

   ```bash
   cp .env.example .env
   # edite .env conforme o ambiente de produção
   ```

2. Execute o script de deploy para construir as imagens, criar os volumes de checkpoints e subir os serviços:

   ```bash
   ./scripts/deploy.sh
   ```

   O script utiliza `docker compose` para inicializar dois containers:

   - **api**: roda `gunicorn` com workers `uvicorn`, carregando as dependências de PyTorch + FastAPI. O volume nomeado `checkpoints` mantém os arquivos de modelo persistentes em `/checkpoints`.
   - **proxy**: container NGINX que faz o proxy reverso (HTTP/HTTPS) para o serviço FastAPI. Monte seus certificados TLS em `./certs` (`server.crt` e `server.key`).

3. Para acompanhar o status dos serviços:

   ```bash
   docker compose logs -f api
   docker compose logs -f proxy
   ```

O endpoint `GET /health` responde com informações sobre o ambiente, tokens configurados e disponibilidade dos checkpoints. Os logs do backend são emitidos em JSON para facilitar integração com plataformas de observabilidade.

## Preparação dos dados

Organize seus exames em diretórios separados por classe, por exemplo:

```
data/
  train/
    normal/
    lesao/
  val/
    normal/
    lesao/
  test/
    normal/
    lesao/
```

Para adicionar outros rótulos (ex.: diferentes tipos de lesão), basta criar novas pastas e ajustar o arquivo de configuração.

## Configuração

O arquivo `configs/default.yaml` define hiperparâmetros básicos:

- caminhos das pastas de treino, validação e teste;
- tamanho do lote, taxa de aprendizado, épocas;
- tamanho das imagens e transformações de normalização.

Você pode criar novas configurações copiando este arquivo e ajustando os parâmetros desejados.

## Treinamento

```bash
python scripts/train.py --config configs/default.yaml
```

O script treina um modelo `ResNet18` pré-treinado e salva os checkpoints em `artifacts/checkpoints`. Métricas e logs ficam disponíveis no console e em um arquivo CSV.

### Interface web interativa

Como alternativa aos scripts de linha de comando, você pode acompanhar todo o fluxo por meio de uma interface construída em Streamlit.

1. Certifique-se de instalar as dependências adicionais (`streamlit`, `pandas`, `plotly`).
2. Inicie a interface:

   ```bash
   streamlit run app/main.py
   ```

3. Use as abas para:
   - **Configuração:** selecionar o arquivo YAML e revisar os parâmetros carregados;
   - **Treinamento:** iniciar o processo diretamente pelo navegador, consultar o arquivo de status (`artifacts/logs/status.json`) e visualizar o histórico de métricas em tempo real;
   - **Inferência:** escolher um checkpoint salvo e enviar uma imagem para obter as probabilidades por classe (incluindo gráfico interativo);
   - **Comparação:** carregar dois exames e calcular SSIM e diferença absoluta média.

O painel lê continuamente os arquivos salvos em `cfg.paths.output_dir` (checkpoints, métricas e `status.json`), permitindo acompanhar o progresso enquanto o treinamento está em execução — seja iniciado pela própria interface ou por outra sessão.

## Inferência

```bash
python scripts/predict.py \
  --config configs/default.yaml \
  --checkpoint artifacts/checkpoints/best.pt \
  --image caminho/para/exame.png
```

A saída inclui a probabilidade prevista para cada classe definida na configuração.

## Comparação de exames

```bash
python scripts/compare.py \
  --reference caminho/para/imagem_normal.png \
  --target caminho/para/imagem_paciente.png
```

O script calcula métricas de similaridade estrutural (SSIM) e diferença absoluta média para auxiliar a inspeção visual.

## Próximos passos sugeridos

- expandir o conjunto de dados com amostras devidamente anonimizadas e rotuladas por especialistas;
- experimentar arquiteturas mais robustas (DenseNet, EfficientNet, Vision Transformers);
- adicionar técnicas de explicabilidade (Grad-CAM) para visualizar regiões relevantes;
- integrar rotinas de validação cruzada e calibração de probabilidades.

## Considerações éticas

- respeite a privacidade e a legislação vigente (LGPD, HIPAA, GDPR) ao manipular dados de pacientes;
- obtenha consentimento apropriado e remova informações identificáveis das imagens;
- envolva profissionais de saúde em todas as etapas de validação e interpretação.

## Licença

Distribuído sob a licença MIT. Consulte `LICENSE` (adicione se necessário).
