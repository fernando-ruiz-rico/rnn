/**
 * Lógica del lado del cliente para la interfaz de predicción de texto.
 * Gestiona los eventos del DOM, implementa un mecanismo de 'debounce' para las 
 * peticiones a la API y renderiza los resultados asíncronos.
 */

// Función de inicialización de los escuchadores de eventos (Event Listeners)
function inicializar() {
  
  // 1. Gestión del evento en el área de texto principal
  document.getElementById('texto').addEventListener('input', function() {
    clearTimeout(timeout); // Reiniciamos el contador si hay nueva actividad

    // Establecemos un retraso de 1000ms (1s) antes de disparar la lógica de negocio
    timeout = setTimeout(() => {
      // Capturamos el valor actual del input de cantidad para enviarlo como parámetro
      cantidad = document.getElementById('cantidad').value;
      
      // 'this.value' hace referencia al contenido actual del textarea
      predecir(this.value, cantidad); 
    }, 1000); 
  });

    // 2. Gestión del evento en el selector de cantidad de palabras
    // Se replica la lógica de debounce para mantener la consistencia si el usuario cambia este valor rápidamente
    document.getElementById('cantidad').addEventListener('input', function() {
      clearTimeout(timeout); 
  
      timeout = setTimeout(() => {
        // En este contexto, necesitamos obtener explícitamente el valor del texto
        texto = document.getElementById('texto').value;
        predecir(texto, this.value); 
      }, 1000); 
    });
}

/**
 * Función asíncrona para solicitar la predicción al backend.
 * * @param {string} texto - La secuencia de texto semilla introducida por el usuario.
 * @param {number} cantidad - Número de palabras a predecir a continuación.
 */
async function predecir(texto, cantidad) {
  const resultado = document.getElementById('resultado'); 

  // Validación básica: evitar peticiones innecesarias si el input está vacío o solo contiene espacios
  if (texto.trim() === "") {
    resultado.innerHTML = "";
    return;
  }

  // Feedback visual para el usuario durante la latencia de la red (UX)
  resultado.innerHTML = `
    <div class="spinner-border text-primary" role="status"></div>
  `;

  try {
    // Petición GET al endpoint del modelo predictivo.
    // Es crítico usar encodeURIComponent para sanear la entrada y evitar errores en la URL con caracteres especiales.
    let respuesta = await fetch(`${URL_PYTHON}/texto_predictivo?texto=${encodeURIComponent(texto)}&cantidad=${cantidad}`);
    
    // Deserialización de la respuesta JSON
    respuesta = await respuesta.json(); 

    // Inyección segura del resultado en el DOM
    // Nota: Se asume que 'respuesta.prediccion' viene saneado del backend o es texto plano seguro.
    resultado.innerHTML = `<p>${respuesta.prediccion}...</p>`;
  } catch (error) {
    // Gestión de errores de red o del servidor para no romper la experiencia de usuario
    console.error("Excepción capturada durante la predicción:", error);
    resultado.innerHTML = "<span class='text-warning'>Error al procesar la solicitud de predicción.</span>";
  }
}

// Punto de entrada: vinculación de eventos al cargar el script
inicializar();