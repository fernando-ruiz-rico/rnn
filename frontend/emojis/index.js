// Función que inicializa el evento de entrada en el campo de texto
function inicializar() {
  // Añade un evento de 'input' al elemento con id 'texto' para detectar cambios en tiempo real
  document.getElementById('texto').addEventListener('input', function() {
    clearTimeout(timeout); // Limpia el temporizador previo si está activo

    // Establece un nuevo temporizador que ejecuta la función traducirEmojis después de 1 segundo
    timeout = setTimeout(() => {
      traducirEmojis(this.value); // Llama a la función que traduce el texto a emojis
    }, 1000); // Espera 1 segundo tras la última pulsación de tecla
  });
}

// Función asíncrona que envía el texto al servidor para ser traducido a emojis
async function traducirEmojis(texto) {
  const resultado = document.getElementById('resultado'); // Referencia al elemento donde se mostrará el resultado

  // Si el texto está vacío, limpia el resultado y detiene la ejecución de la función
  if (texto.trim() === "") {
    resultado.innerHTML = "";
    return;
  }

  try {
    // Construye la URL con los parámetros GET, codificando el texto para que sea seguro enviar en la URL
    let respuesta = await fetch(`${URL_PYTHON}/emojis?texto=${encodeURIComponent(texto)}`);
    respuesta = await respuesta.json(); // Convierte la respuesta en formato JSON

    // Asumiendo que la respuesta es un array de objetos con propiedades "emojis" y "similitud"
    // Muestra cada traducción de emoji en un nuevo párrafo
    if (respuesta.length) {
      resultado.innerHTML = respuesta.map(item => `<p>${item.emojis}</p>`).join('');
    }
    else {
      // Si no se recibe una respuesta válida, muestra un mensaje de advertencia
      resultado.innerHTML = "<span class='text-warning'>No se han encontrado emojis</span>";
    }

  } catch (error) {
    // Muestra un mensaje de error si la solicitud falla
    console.error("Error al traducir a emojis:", error);
    resultado.innerHTML = "<span class='text-warning'>Error al traducir el texto</span>";
  }
}

// Llama a la función inicializar para configurar los eventos del formulario
inicializar();
