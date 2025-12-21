// Diccionario maestro de empresas mapeado a sus tickers (símbolos) compatibles con Yahoo Finance.
// Separado por geografía para organizar los optgroups en el frontend.
const empresas = {
  "Empresas de Estados Unidos": {
    "3M": "MMM",
    "AbbVie": "ABBV",
    "Adobe": "ADBE",
    "AMD": "AMD",
    "Amazon": "AMZN",
    "Apple": "AAPL",
    "AT&T": "T",
    "Bank of America": "BAC",
    "Berkshire Hathaway": "BRK-B",
    "Boeing": "BA",
    "Broadcom": "AVGO",
    "Caterpillar": "CAT",
    "Chevron": "CVX",
    "Cisco Systems": "CSCO",
    "Coca-Cola": "KO",
    "Comcast": "CMCSA",
    "Costco": "COST",
    "Disney": "DIS",
    "ExxonMobil": "XOM",
    "Goldman Sachs": "GS",
    "Google": "GOOGL",
    "Home Depot": "HD",
    "Honeywell": "HON",
    "IBM": "IBM",
    "Intel": "INTC",
    "Johnson & Johnson": "JNJ",
    "JPMorgan Chase": "JPM",
    "Lowe's": "LOW",
    "Mastercard": "MA",
    "McDonald's": "MCD",
    "Merck & Co.": "MRK",
    "Meta Platforms Inc. (Facebook)": "META",
    "Microsoft": "MSFT",
    "Netflix": "NFLX",
    "Nike": "NKE",
    "Nvidia": "NVDA",
    "Oracle": "ORCL",
    "PayPal": "PYPL",
    "PepsiCo": "PEP",
    "Pfizer": "PFE",
    "Procter & Gamble": "PG",
    "Qualcomm": "QCOM",
    "Salesforce": "CRM",
    "Starbucks": "SBUX",
    "Target": "TGT",
    "Tesla": "TSLA",
    "Texas Instruments": "TXN",
    "UnitedHealth Group": "UNH",
    "United Parcel Service": "UPS",
    "Verizon Communications": "VZ",
    "Visa": "V",
    "Walmart": "WMT",
    "Wells Fargo": "WFC"
  },
  "Empresas de España": {
    "Acciona": "ANA.MC",
    "ACS": "ACS.MC",
    "Amadeus IT Group": "AMS.MC",
    "BBVA": "BBVA.MC",
    "Banco Santander": "SAN.MC",
    "CaixaBank": "CABK.MC",
    "Cellnex Telecom": "CLNX.MC",
    "Colonial": "COL.MC",
    "Endesa": "ELE.MC",
    "Ferrovial": "FER.MC",
    "Grifols": "GRF.MC",
    "Iberdrola": "IBE.MC",
    "Inditex": "ITX.MC",
    "Mapfre": "MAP.MC",
    "Merlin Properties": "MRL.MC",
    "Naturgy": "NTGY.MC",
    "Red Eléctrica": "REE",
    "Repsol": "REP.MC",
    "Siemens Energy": "ENR",
    "Telefónica": "TEF.MC"
  }
};

/**
 * Genera dinámicamente el elemento <select> en el DOM basado en el objeto 'empresas'.
 * Utiliza <optgroup> para mantener la jerarquía visual por país.
 */
function crearSelect() {
  const select = document.getElementById('empresa'); // Referencia al nodo del DOM

  // Placeholder inicial para forzar una selección consciente del usuario
  select.innerHTML = `<option value="" selected disabled>Selecciona una empresa...</option>`;

  // Iteración sobre claves geográficas
  for (grupo in empresas) {
    select.innerHTML += `<optgroup label="${grupo}">`; 
    for (empresa in empresas[grupo]) {
      const simbolo = empresas[grupo][empresa]; 
      // Inyección del valor (ticker) y texto visible (nombre humano)
      select.innerHTML += `<option value="${simbolo}">${empresa}</option>`; 
    }
    select.innerHTML += `</optgroup>`; 
  }

  // Listener para disparar la petición en cuanto cambia el valor
  select.addEventListener('change', function() {
    obtenerGrafica(this.value); 
  });
}

/**
 * Gestiona la petición asíncrona de la imagen generada por Python.
 * Incluye gestión de estado de carga (loading spinner) dado que el entrenamiento
 * de la red neuronal en backend provoca latencia.
 * @param {string} empresa - Ticker de la empresa (ej: 'AAPL')
 */
function obtenerGrafica(empresa) {
  const finanzas = document.getElementById('finanzas');
  
  // Feedback inmediato al usuario mediante Spinner de Bootstrap
  finanzas.innerHTML = `
    <div class="fs-5 text-center">
      <p>Realizando predicción precios de acciones para el próximo año.<br>
         Esto puede tardar unos minutos...</p>
      <div class="spinner-border" role="status">
        <span class="visually-hidden">Cargando...</span>
      </div>
    </div>`;
  
  // Uso del objeto Image para precarga y gestión de eventos onload/onerror
  const img = new Image();
  // Asumimos que URL_PYTHON está definida globalmente o inyectada en el entorno
  img.src = `${URL_PYTHON}/finanzas_prediccion/grafica?empresa=${empresa}`;
  img.className = "img-fluid";
  
  // Callback: Reemplazo del spinner solo cuando la imagen está totalmente recibida
  img.onload = function() {
    finanzas.innerHTML = ''; // Limpieza del contenedor
    finanzas.appendChild(img); // Inserción limpia
  };
  
  // Gestión básica de errores de red o servidor
  img.onerror = function() {
    finanzas.innerHTML = '<div class="alert alert-danger">Error al cargar la imagen. Inténtelo de nuevo.</div>';
  };
}


// Punto de entrada: Inicialización de componentes al cargar el script
function inicializa() {
  crearSelect(); 
}

inicializa();