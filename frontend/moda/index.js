// Inicialización del controlador principal
function inicializar() {
  // Disparar la primera carga automática al montar el componente
  cargarNuevaModa();

  // Binding del evento de generación manual
  document.getElementById('btn-generar').addEventListener('click', function() {
    cargarNuevaModa();
  });
}

/**
 * Solicita una nueva imagen generada al backend y actualiza el DOM.
 * Maneja el flujo de estados de la UI (Cargando -> Éxito/Error).
 */
async function cargarNuevaModa() {
  const imgElement = document.getElementById('imagen-moda');
  const spinner = document.getElementById('spinner-carga');
  const btnGenerar = document.getElementById('btn-generar');
  const mensajeDiv = document.getElementById('mensaje-estado');

  // --- GESTIÓN DE ESTADO UI (Pre-fetch) ---
  imgElement.style.display = 'none';   // Ocultar imagen obsoleta
  spinner.style.display = 'block';     // Feedback visual de carga
  btnGenerar.disabled = true;          // Prevención de condiciones de carrera (doble submit)
  mensajeDiv.innerHTML = "Consultando a la IA...";

  try {
    // Petición al endpoint. Nota: Se espera una respuesta binaria (image/png), no JSON.
    // URL_PYTHON debe estar definida globalmente o importada.
    let respuesta = await fetch(`${URL_PYTHON}/moda`);

    if (!respuesta.ok) {
        throw new Error(`Error del servidor: ${respuesta.status}`);
    }

    // Procesamiento de flujo binario:
    // Convertimos el stream de respuesta en un Blob inmutable.
    let blob = await respuesta.blob();
    
    // Generamos una URL de objeto temporal en memoria (blob:http://...)
    // Esto permite al tag <img> acceder a los datos binarios sin guardarlos en disco local.
    let imagenUrl = URL.createObjectURL(blob);

    // Actualización del Source de la imagen
    imgElement.src = imagenUrl;

    // Listener 'onload': Esperamos a que el motor de renderizado del navegador
    // haya procesado los datos binarios antes de restaurar la interfaz.
    imgElement.onload = () => {
        spinner.style.display = 'none';
        imgElement.style.display = 'block';
        btnGenerar.disabled = false;
        mensajeDiv.innerHTML = "";
        
        // Opcional: Aquí se podría revocar la URL del objeto (URL.revokeObjectURL) 
        // si la gestión de memoria fuera crítica en una SPA de larga duración.
    };

  } catch (error) {
    // Manejo robusto de errores de red o del backend
    console.error("Error al cargar moda:", error);
    spinner.style.display = 'none';
    btnGenerar.disabled = false;
    mensajeDiv.innerHTML = "<span class='text-danger'>⚠️ Error al generar la imagen. Intenta de nuevo.</span>";
  }
}

// Punto de entrada
inicializar();