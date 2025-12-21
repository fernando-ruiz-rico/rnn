// Variables globales de estado
let solucionCorrecta = []; // Almacena la secuencia correcta de prendas (backend)
let intentosRestantes = 5;
const MAX_INTENTOS = 5;

// Inicialización de listeners de eventos al cargar el script
function inicializar() {
  cargarNuevoCaptcha();

  // Listener para click en verificar
  document.getElementById('btn-verificar').addEventListener('click', function() {
    const intento = document.getElementById('intento-usuario').value;
    verificarCaptcha(intento);
  });

  // Listener para recargar nuevo reto
  document.getElementById('btn-recargar').addEventListener('click', function() {
    cargarNuevoCaptcha();
  });
  
  // Usabilidad: Permitir enviar con la tecla Enter
  document.getElementById('intento-usuario').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
      const intento = this.value;
      verificarCaptcha(intento);
    }
  });
}

/**
 * Función asíncrona para obtener un nuevo reto del backend.
 * Gestiona el spinner de carga y resetea el estado del juego.
 */
async function cargarNuevoCaptcha() {
  const imgElement = document.getElementById('imagen-captcha');
  const spinner = document.getElementById('spinner-carga');
  const resultadoDiv = document.getElementById('resultado');
  const inputUsuario = document.getElementById('intento-usuario');
  const btnVerificar = document.getElementById('btn-verificar');

  // --- REINICIO DE ESTADO UI ---
  intentosRestantes = MAX_INTENTOS;
  inputUsuario.value = "";
  inputUsuario.disabled = false;
  btnVerificar.disabled = false;
  inputUsuario.focus(); // Foco automático para mejorar UX
  resultadoDiv.innerHTML = "";
  
  actualizarBadgeIntentos();

  // Gestión visual de la carga
  imgElement.style.display = 'none';
  spinner.style.display = 'block';

  try {
    // Petición al endpoint V2 (Python)
    let respuesta = await fetch(`${URL_PYTHON}/captcha_v2`); 
    let data = await respuesta.json();

    if (data.error) throw new Error(data.error);

    // Inyección de la imagen en Base64 y almacenamiento de la solución
    imgElement.src = "data:image/png;base64," + data.img;
    solucionCorrecta = data.solucion; // Se espera un array: ["Camiseta", "Bota", ...]
    
    // Debug: útil en desarrollo, debe eliminarse o comentarse en producción
    console.log("Solución (Debug):", solucionCorrecta);

    // Solo mostramos la imagen cuando el navegador ha terminado de renderizarla
    imgElement.onload = () => {
        spinner.style.display = 'none';
        imgElement.style.display = 'block';
    };

  } catch (error) {
    console.error("Error crítico:", error);
    spinner.style.display = 'none';
    resultadoDiv.innerHTML = "<span class='text-danger'>Error de conexión con el Generador de Moda</span>";
  }
}

/**
 * Lógica de validación.
 * Compara la entrada del usuario con la solución del servidor.
 * Incluye normalización para ser tolerante a formatos.
 */
function verificarCaptcha(intento) {
  const resultadoDiv = document.getElementById('resultado');
  const inputUsuario = document.getElementById('intento-usuario');
  const btnVerificar = document.getElementById('btn-verificar');

  if (!intento || intento.trim() === "") return;

  // 1. Normalización de entrada del usuario
  // Convertimos a minúsculas, separamos por delimitadores comunes (espacio o coma)
  // y filtramos elementos vacíos para evitar errores de longitud.
  const palabrasUsuario = intento.toLowerCase()
    .split(/[\s,]+/) 
    .filter(p => p.length > 0);

  // 2. Normalización de la solución correcta (backend)
  const palabrasSolucion = solucionCorrecta.map(s => s.toLowerCase());

  // 3. Comparación estricta de arrays (posición y valor)
  let esCorrecto = true;
  if (palabrasUsuario.length !== palabrasSolucion.length) {
      esCorrecto = false; // Longitud incorrecta implica fallo inmediato
  } else {
      for (let i = 0; i < palabrasSolucion.length; i++) {
          if (palabrasUsuario[i] !== palabrasSolucion[i]) {
              esCorrecto = false;
              break;
          }
      }
  }

  // --- ESCENARIO 1: ACIERTO ---
  if (esCorrecto) {
    resultadoDiv.innerHTML = `
        <span class='text-success'>✅ ¡CORRECTO!</span>
        <div class='fs-6 text-muted'>Has identificado las prendas.</div>
    `;
    // Bloqueamos controles para evitar envíos posteriores al éxito
    inputUsuario.disabled = true;
    btnVerificar.disabled = true;
    return;
  } 

  // --- ESCENARIO 2: FALLO ---
  intentosRestantes--;
  actualizarBadgeIntentos();

  if (intentosRestantes > 0) {
    // Feedback de error pero permitiendo reintento
    resultadoDiv.innerHTML = `
        <span class='text-warning fs-4'>❌ Incorrecto</span>
        <div class='fs-6 text-muted'>Revisa el orden y la ortografía.</div>
    `;
  } else {
    // --- ESCENARIO 3: GAME OVER ---
    inputUsuario.disabled = true;
    btnVerificar.disabled = true;
    
    // Mostramos la solución formateada para que el usuario aprenda
    const solucionTexto = solucionCorrecta.join(", ");
    
    resultadoDiv.innerHTML = `
        <span class='text-danger'>⛔ Fin del juego</span>
        <div class='fs-5 text-primary mt-2'>Eran: <b>${solucionTexto}</b></div>
        <div class='fs-6 text-muted'>Pulsa "Generar nuevo" para reintentar.</div>
    `;
  }
}

// Actualiza el indicador visual de intentos (Badge de Bootstrap)
function actualizarBadgeIntentos() {
    const badge = document.getElementById('badge-intentos');
    badge.innerText = `Intentos restantes: ${intentosRestantes}`;
    
    badge.className = 'badge mb-2 fs-6 ';
    
    // Cambio de color semántico según urgencia
    if (intentosRestantes === 3) badge.classList.add('bg-success');
    else if (intentosRestantes === 2) badge.classList.add('bg-warning', 'text-dark');
    else if (intentosRestantes === 1) badge.classList.add('bg-danger');
    else badge.classList.add('bg-secondary');
}

// Arranque
inicializar();