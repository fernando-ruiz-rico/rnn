// Inicialización
function inicializar() {
  // Cargar la primera imagen al entrar
  cargarNuevaModa();

  // Configurar el botón
  document.getElementById('btn-generar').addEventListener('click', function() {
    cargarNuevaModa();
  });
}

// Carga imagen desde el backend
async function cargarNuevaModa() {
  const imgElement = document.getElementById('imagen-moda');
  const spinner = document.getElementById('spinner-carga');
  const btnGenerar = document.getElementById('btn-generar');
  const mensajeDiv = document.getElementById('mensaje-estado');

  // --- INTERFAZ DE CARGA ---
  imgElement.style.display = 'none';   // Ocultar imagen anterior
  spinner.style.display = 'block';     // Mostrar spinner
  btnGenerar.disabled = true;          // Evitar doble clic
  mensajeDiv.innerHTML = "Consultando a la IA...";

  try {
    // Nota: Como tu backend hace 'send_file', recibimos una imagen binaria (Blob), no JSON.
    // Usamos URL_PYTHON de variables.js, concatenando tu ruta '/api/moda'
    let respuesta = await fetch(`${URL_PYTHON}/moda`);

    if (!respuesta.ok) {
        throw new Error(`Error del servidor: ${respuesta.status}`);
    }

    // Convertimos la respuesta en un Blob (objeto binario de imagen)
    let blob = await respuesta.blob();
    
    // Creamos una URL temporal para ese blob
    let imagenUrl = URL.createObjectURL(blob);

    // Asignamos la URL a la imagen
    imgElement.src = imagenUrl;

    // Cuando la imagen termine de cargar en el navegador:
    imgElement.onload = () => {
        spinner.style.display = 'none';
        imgElement.style.display = 'block';
        btnGenerar.disabled = false;
        mensajeDiv.innerHTML = "";
    };

  } catch (error) {
    console.error("Error al cargar moda:", error);
    spinner.style.display = 'none';
    btnGenerar.disabled = false;
    mensajeDiv.innerHTML = "<span class='text-danger'>⚠️ Error al generar la imagen. Intenta de nuevo.</span>";
  }
}

// Arrancar script
inicializar();