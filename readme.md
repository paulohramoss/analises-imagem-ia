# Plataforma de Análise de Imagens Médicas (Prototipo de Pesquisa)

> **Aviso importante:** este projeto é apenas um protótipo de pesquisa e não deve ser usado para emitir diagnósticos clínicos ou substituir a avaliação de profissionais de saúde. Os modelos produzidos com este código precisam passar por validações clínicas, revisão ética e aprovação regulatória antes de qualquer uso real.

## Visão geral

Este repositório fornece uma estrutura em Python para treinar e avaliar modelos de classificação de imagens médicas (por exemplo, ressonâncias magnéticas) utilizando `PyTorch`. O objetivo é detectar padrões associados a lesões comparando exames de pacientes com imagens de referência consideradas "normais".

O projeto inclui:

- scripts de treinamento, inferência e comparação de exames;
- pipeline de dados baseado em diretórios rotulados (`normal`, `lesao`, etc.);
- configuração por arquivos YAML;
- funções utilitárias para registro e checkpoints.

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
├── requirements.txt
├── scripts/
│   ├── compare.py
│   ├── predict.py
│   └── train.py
└── src/
    └── medimaging_ai/
        ├── __init__.py
        ├── compare.py
        ├── config.py
        ├── data.py
        ├── inference.py
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

## Integração com WhatsApp e persistência dos resultados

- Utilize `scripts/whatsapp_webhook.py` para expor um endpoint FastAPI capaz de receber webhooks (ex.: Twilio WhatsApp).
- Cada requisição registrada gera uma entrada na tabela `analises` (`sqlite`) com metadados, probabilidades calculadas pelo modelo e carimbo de tempo.
- Os arquivos temporários usados durante a inferência são eliminados automaticamente; opcionalmente é possível manter uma cópia em um diretório seguro configurado via variável de ambiente.

Inicie o servidor:

```bash
uvicorn scripts.whatsapp_webhook:app --host 0.0.0.0 --port 8000
```

### Variáveis de ambiente

Configure as variáveis abaixo para controlar o comportamento do webhook:

- `WHATSAPP_CONFIG_PATH`: caminho para o arquivo YAML de configuração do experimento (obrigatório).
- `WHATSAPP_CHECKPOINT_PATH`: checkpoint `.pt` utilizado na inferência (obrigatório).
- `WHATSAPP_DB_PATH`: caminho do banco SQLite que armazenará a tabela `analises` (padrão: `artifacts/whatsapp.sqlite`).
- `WHATSAPP_PROVIDER_TOKEN`: token/bearer utilizado para baixar mídias do provedor.
- `WHATSAPP_PROVIDER_BASE_URL`: URL base para compor o download da mídia quando o provedor retorna caminhos relativos.
- `WHATSAPP_RETAIN_MEDIA`: defina como `true`/`1` para manter uma cópia do arquivo analisado após a inferência.
- `WHATSAPP_RETENTION_DIR`: diretório seguro onde os arquivos serão armazenados caso `WHATSAPP_RETAIN_MEDIA` esteja ativo.

## Políticas de privacidade e consentimento

- Somente armazene mídias e metadados após garantir consentimento explícito do paciente/usuário e justifique o propósito em termos claros.
- Garanta que o diretório/armazenamento configurado em `WHATSAPP_RETENTION_DIR` possua controles de acesso equivalentes a um "bucket" seguro (criptografia em repouso, controle de chaves e auditoria).
- Exclua periodicamente os registros que não forem mais necessários e respeite pedidos de anonimização, utilizando o identificador único (`message_id`) como referência.
- Atualize contratos, termos de uso e comunicações com pacientes para refletir o fluxo de dados descrito nesta documentação.

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
