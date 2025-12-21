// Objeto maestro de configuración.
// Mapea el nombre legible de la empresa con su Ticker oficial en Yahoo Finance.
// Se ha estructurado en niveles para permitir la creación de <optgroup> en el HTML.
const empresas = {
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
  },
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
  }
};

// Función encargada de poblar dinámicamente el selector del DOM.
function crearSelect() {
  const select = document.getElementById('empresa'); // Referencia al elemento <select> en el DOM

  // Establece una opción por defecto deshabilitada para forzar una elección consciente del usuario
  select.innerHTML = `<option value="" selected disabled>Selecciona una empresa...</option>`;

  // Recorrido jerárquico del objeto 'empresas'.
  // Primer nivel: Región/País -> Crea <optgroup>
  for (grupo in empresas) {
    select.innerHTML += `<optgroup label="${grupo}">`; 
    // Segundo nivel: Empresa -> Crea <option>
    for (empresa in empresas[grupo]) {
      const simbolo = empresas[grupo][empresa]; // Obtiene el símbolo bursátil de la empresa
      select.innerHTML += `<option value="${simbolo}">${empresa}</option>`; // Añade cada empresa como una opción del select
    }
    select.innerHTML += `</optgroup>`; // Cierre correcto de la etiqueta de grupo
  }

  // Listener para disparar la petición en cuanto el usuario modifica el valor del selector
  select.addEventListener('change', function() {
    obtenerGrafica(this.value); // 'this.value' contiene el ticker (ej: SAN.MC)
  });
}

// Función que solicita la gráfica financiera de la empresa seleccionada y la muestra
function obtenerGrafica(empresa) {  
  finanzas = document.getElementById('finanzas'); // Referencia al contenedor de visualización

  // Inyección directa de HTML.
  // IMPORTANTE: La URL apunta a un endpoint de Python que devuelve un stream de bytes (imagen PNG).
  // No se descarga un fichero, se renderiza directamente en el navegador.
  finanzas.innerHTML = `<img src="${URL_PYTHON}/finanzas/grafica?empresa=${empresa}" class="img-fluid">`;
}

// Función de arranque (entry point) para garantizar que el DOM esté listo antes de manipularlo
function inicializa() {
  crearSelect(); // Llama a la función para crear el <select> de empresas
}

// Ejecución inmediata al cargar el script
inicializa();