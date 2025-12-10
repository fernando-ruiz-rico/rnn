// Objeto que almacena los nombres de las empresas y sus símbolos bursátiles
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

// Función que crea el elemento <select> con las opciones de empresas y sus respectivos símbolos
function crearSelect() {
  const select = document.getElementById('empresa'); // Referencia al elemento <select> en el DOM

  // Establece una opción por defecto deshabilitada
  select.innerHTML = `<option value="" selected disabled>Selecciona una empresa...</option>`;

  // Itera sobre los grupos de empresas (por país)
  for (grupo in empresas) {
    select.innerHTML += `<optgroup label="${grupo}">`; // Crea un grupo de opciones (<optgroup>) para cada país
    for (empresa in empresas[grupo]) {
      const simbolo = empresas[grupo][empresa]; // Obtiene el símbolo bursátil de la empresa
      select.innerHTML += `<option value="${simbolo}">${empresa}</option>`; // Añade cada empresa como una opción del select
    }
    select.innerHTML += `</optgroup>`; // Cierra el grupo de opciones
  }

  // Añade un evento para detectar cambios en la selección del <select>
  select.addEventListener('change', function() {
    obtenerGrafica(this.value); // Llama a la función para obtener la gráfica financiera
  });
}

// Función que solicita la gráfica financiera de la empresa seleccionada y la muestra
function obtenerGrafica(empresa) {  
  finanzas = document.getElementById('finanzas'); // Referencia al elemento donde se mostrará la gráfica

  // Establece el contenido del elemento como una imagen (gráfica) obtenida desde la URL del servidor Python
  finanzas.innerHTML = `<img src="${URL_PYTHON}/finanzas/grafica?empresa=${empresa}" class="img-fluid">`;
}

// Función que inicializa la aplicación creando el select de empresas
function inicializa() {
  crearSelect(); // Llama a la función para crear el <select> de empresas
}

// Llama a la función de inicialización cuando se carga el script
inicializa();
