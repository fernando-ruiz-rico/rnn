// Variables globales
let solucionCorrecta = "";
let intentosRestantes = 3;
const MAX_INTENTOS = 3;

// Inicialización
function inicializar() {
  cargarNuevoCaptcha();

  document.getElementById('btn-verificar').addEventListener('click', function() {
    const intento = document.getElementById('intento-usuario').value;
    verificarCaptcha(intento);
  });

  document.getElementById('btn-recargar').addEventListener('click', function() {
    cargarNuevoCaptcha();
  });
  
  // Permitir verificar con la tecla Enter
  document.getElementById('intento-usuario').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
      const intento = this.value;
      verificarCaptcha(intento);
    }
  });
}

// Carga imagen y resetea contadores
async function cargarNuevoCaptcha() {
  const imgElement = document.getElementById('imagen-captcha');
  const spinner = document.getElementById('spinner-carga');
  const resultadoDiv = document.getElementById('resultado');
  const inputUsuario = document.getElementById('intento-usuario');
  const btnVerificar = document.getElementById('btn-verificar');
  const badgeIntentos = document.getElementById('badge-intentos');

  // --- RESETER ESTADO ---
  intentosRestantes = MAX_INTENTOS; // Volvemos a 3 intentos
  inputUsuario.value = "";
  inputUsuario.disabled = false;    // Habilitamos input
  btnVerificar.disabled = false;    // Habilitamos botón
  inputUsuario.focus();
  resultadoDiv.innerHTML = "";
  
  // Actualizar visual del contador
  actualizarBadgeIntentos();

  // Interfaz de carga
  imgElement.style.display = 'none';
  spinner.style.display = 'block';

  try {
    let respuesta = await fetch(`${URL_PYTHON}/captcha`);
    let data = await respuesta.json();

    if (data.error) throw new Error(data.error);

    imgElement.src = "data:image/png;base64," + data.imagen_base64;
    solucionCorrecta = data.solucion_real;
    
    // Debug
    console.log("Solución (Debug):", solucionCorrecta);

    imgElement.onload = () => {
        spinner.style.display = 'none';
        imgElement.style.display = 'block';
    };

  } catch (error) {
    console.error("Error:", error);
    spinner.style.display = 'none';
    resultadoDiv.innerHTML = "<span class='text-danger'>Error de conexión</span>";
  }
}

// Lógica de verificación
function verificarCaptcha(intento) {
  const resultadoDiv = document.getElementById('resultado');
  const inputUsuario = document.getElementById('intento-usuario');
  const btnVerificar = document.getElementById('btn-verificar');

  // Validar que haya escrito algo
  if (!intento || intento.trim() === "") {
     return; // No hacemos nada si está vacío
  }

  // --- CASO 1: ACIERTO ---
  if (intento.trim() === solucionCorrecta.trim()) {
    resultadoDiv.innerHTML = `
        <span class='text-success'>✅ ¡CORRECTO!</span>
        <div class='fs-6 text-muted'>Eres humano.</div>
    `;
    inputUsuario.disabled = true; // Bloqueamos para que no siga jugando
    btnVerificar.disabled = true;
    return;
  } 

  // --- CASO 2: FALLO ---
  // Restamos un intento
  intentosRestantes--;
  actualizarBadgeIntentos(); // Actualizamos el color y texto del badge

  if (intentosRestantes > 0) {
    // Todavía le quedan vidas
    inputUsuario.value = ""; // Borramos lo que escribió para que intente de nuevo
    inputUsuario.focus();
    resultadoDiv.innerHTML = `
        <span class='text-warning fs-4'>❌ Incorrecto</span>
        <div class='fs-6 text-muted'>Inténtalo de nuevo.</div>
    `;
  } else {
    // --- CASO 3: GAME OVER (0 intentos) ---
    inputUsuario.disabled = true;
    btnVerificar.disabled = true;
    
    resultadoDiv.innerHTML = `
        <span class='text-danger'>⛔ Se acabaron los intentos</span>
        <div class='fs-4 text-primary mt-2'>La solución era: <b>${solucionCorrecta}</b></div>
        <div class='fs-6 text-muted'>Pulsa "Generar nuevo" para reintentar.</div>
    `;
  }
}

// Función auxiliar para colorear el badge según los intentos
function actualizarBadgeIntentos() {
    const badge = document.getElementById('badge-intentos');
    badge.innerText = `Intentos restantes: ${intentosRestantes}`;
    
    // Cambiar colores dinámicamente
    badge.className = 'badge mb-2 fs-6 '; // Clases base
    
    if (intentosRestantes === 3) {
        badge.classList.add('bg-success'); // Verde
    } else if (intentosRestantes === 2) {
        badge.classList.add('bg-warning', 'text-dark'); // Amarillo
    } else if (intentosRestantes === 1) {
        badge.classList.add('bg-danger'); // Rojo
    } else {
        badge.classList.add('bg-secondary'); // Gris (0 intentos)
    }
}

inicializar();