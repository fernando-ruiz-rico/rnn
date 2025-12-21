// Función que inicializa el evento de entrada en el campo de texto.
// Configura los listeners necesarios para la interacción del usuario.
function inicializar() {
  // Añade un evento de 'input' al elemento con id 'texto' para detectar cambios en tiempo real.
  document.getElementById('texto').addEventListener('input', function() {
    
    // Implementación de DEBOUNCE:
    // Limpia el temporizador previo si el usuario sigue escribiendo. Esto evita realizar
    // una petición HTTP por cada carácter pulsado, optimizando el rendimiento del servidor.
    clearTimeout(timeout); 

    // Establece un nuevo temporizador que ejecuta la función traducirEmojis después de 1 segundo de inactividad.
    timeout = setTimeout(() => {
      // Usamos una función flecha para mantener el contexto o pasamos el valor directamente.
      traducirEmojis(this.value); 
    }, 1000); 
  });
}

// Función asíncrona que gestiona la comunicación con la API Python.
async function traducirEmojis(texto) {
  const resultado = document.getElementById('resultado'); // Referencia al contenedor del DOM.

  // Validación básica: Si el texto está vacío, limpiamos la interfaz y abortamos la petición.
  if (texto.trim() === "") {
    resultado.innerHTML = "";
    return;
  }

  try {
    // Petición GET al endpoint del backend.
    // Es crítico usar encodeURIComponent para sanear la entrada y evitar errores con caracteres especiales en la URL.
    let respuesta = await fetch(`${URL_PYTHON}/emojis?texto=${encodeURIComponent(texto)}`);
    
    // Parseo de la respuesta asíncrona a JSON.
    respuesta = await respuesta.json();

    // Renderizado condicional basado en la respuesta del modelo.
    // Se asume que el backend devuelve una lista de objetos {emojis: string, similitud: number}.
    if (respuesta.length) {
      // Transformación de datos a HTML mediante map y join para una inserción eficiente en el DOM.
      resultado.innerHTML = respuesta.map(item => `<p>${item.emojis}</p>`).join('');
    }
    else {
      // Feedback al usuario en caso de que el modelo no encuentre coincidencias con el umbral definido.
      resultado.innerHTML = "<span class='text-warning'>No se han encontrado emojis</span>";
    }

  } catch (error) {
    // Gestión de errores de red o del servidor para evitar que la aplicación se rompa silenciosamente.
    console.error("Error al traducir a emojis:", error);
    resultado.innerHTML = "<span class='text-warning'>Error al traducir el texto</span>";
  }
}

// Punto de entrada de la lógica de frontend.
inicializar();