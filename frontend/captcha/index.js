/**
 * index.js
 * Lógica del frontend para la gestión del ciclo de vida del Captcha.
 *
 * NOTA DE ARQUITECTURA:
 * La validación se está realizando en el lado del cliente (Client-Side Validation) por simplicidad
 * en este entorno de demostración. En un entorno de producción estricto, la validación debería
 * realizarse siempre en el servidor para evitar la manipulación de la variable 'solucionCorrecta'.
 */

// --- ESTADO GLOBAL ---

// Almacenamiento temporal de la solución hash/texto proveniente del backend.
// [SEGURIDAD] Expuesto en memoria del navegador. Vulnerable a inspección si no se ofusca o valida en servidor.
let solucionCorrecta = "";

// Gestión de intentos del usuario.
let intentosRestantes = 3;
const MAX_INTENTOS = 3;

// --- INICIALIZACIÓN ---

/**
 * Función de arranque.
 * Configura los listeners de eventos y dispara la primera carga de datos.
 */
function inicializar() {
  cargarNuevoCaptcha();

  // Asignación de evento para la verificación manual mediante clic.
  document.getElementById('btn-verificar').addEventListener('click', function() {
    const intento = document.getElementById('intento-usuario').value;
    verificarCaptcha(intento);
  });

  // Listener para refrescar el captcha en casos de ilegibilidad o bloqueo.
  document.getElementById('btn-recargar').addEventListener('click', function() {
    cargarNuevoCaptcha();
  });
  
  // Mejora de UX: Permitir el envío del formulario mediante la tecla Enter.
  document.getElementById('intento-usuario').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
      const intento = this.value;
      verificarCaptcha(intento);
    }
  });
}

// --- LÓGICA DE RED Y DOM ---

/**
 * Solicita un nuevo captcha al microservicio de Python.
 * Gestiona el estado de carga (spinners) y reinicia los contadores de juego.
 */
async function cargarNuevoCaptcha() {
  // Referencias a elementos del DOM para manipulación directa.
  const imgElement = document.getElementById('imagen-captcha');
  const spinner = document.getElementById('spinner-carga');
  const resultadoDiv = document.getElementById('resultado');
  const inputUsuario = document.getElementById('intento-usuario');
  const btnVerificar = document.getElementById('btn-verificar');
  const badgeIntentos = document.getElementById('badge-intentos');

  // [RESET DE ESTADO]
  // Es crítico reiniciar las condiciones de victoria/derrota al cargar una nueva instancia.
  intentosRestantes = MAX_INTENTOS;
  inputUsuario.value = "";
  inputUsuario.disabled = false;    // Reactivación de inputs tras un posible bloqueo anterior.
  btnVerificar.disabled = false;    
  inputUsuario.focus();             // Foco automático para mejorar la accesibilidad y velocidad.
  resultadoDiv.innerHTML = "";
  
  // Sincronización visual del contador de intentos.
  actualizarBadgeIntentos();

  // Gestión visual de la carga asíncrona (UX).
  imgElement.style.display = 'none';
  spinner.style.display = 'block';

  try {
    // Petición asíncrona al endpoint de generación.
    // Se espera un JSON con la imagen en base64 y la solución.
    let respuesta = await fetch(`${URL_PYTHON}/captcha`);
    let data = await respuesta.json();

    if (data.error) throw new Error(data.error);

    // Inyección directa del stream base64 en el source de la imagen.
    imgElement.src = "data:image/png;base64," + data.img;
    solucionCorrecta = data.solucion;
    
    // [DEBUG] Solo visible en entornos de desarrollo.
    // Permite verificar la integridad de la red neuronal generadora sin resolver el puzzle visualmente.
    console.log("Solución (Debug):", solucionCorrecta);

    // Evento onload: Aseguramos que el spinner solo desaparezca cuando la imagen esté renderizada.
    imgElement.onload = () => {
        spinner.style.display = 'none';
        imgElement.style.display = 'block';
    };

  } catch (error) {
    // Gestión de errores de red o del backend.
    console.error("Excepción capturada en fetch:", error);
    spinner.style.display = 'none';
    resultadoDiv.innerHTML = "<span class='text-danger'>Error de conexión con el servicio de IA</span>";
  }
}

// --- LÓGICA DE CAPTCHA ---

/**
 * Compara el input del usuario con la solución almacenada.
 * Gestiona las transiciones de estado (Ganar, Perder Vida, Game Over).
 * @param {string} intento - Valor introducido por el usuario.
 */
function verificarCaptcha(intento) {
  const resultadoDiv = document.getElementById('resultado');
  const inputUsuario = document.getElementById('intento-usuario');
  const btnVerificar = document.getElementById('btn-verificar');

  // Validación básica de entrada: Evitar penalización por inputs vacíos.
  if (!intento || intento.trim() === "") {
     return; 
  }

  // ESCENARIO 1: ACIERTO
  // Se normalizan ambas cadenas (trim) para evitar errores por espacios accidentales.
  if (intento.trim() === solucionCorrecta.trim()) {
    resultadoDiv.innerHTML = `
        <span class='text-success'>✅ ¡CORRECTO!</span>
        <div class='fs-6 text-muted'>Validación biométrica (simulada) exitosa.</div>
    `;
    // Bloqueo de controles para finalizar el flujo.
    inputUsuario.disabled = true; 
    btnVerificar.disabled = true;
    return;
  } 

  // ESCENARIO 2: FALLO
  intentosRestantes--;
  actualizarBadgeIntentos(); 

  if (intentosRestantes > 0) {
    // Flujo de reintento: Limpiamos input y mantenemos foco.
    inputUsuario.value = ""; 
    inputUsuario.focus();
    resultadoDiv.innerHTML = `
        <span class='text-warning fs-4'>❌ Incorrecto</span>
        <div class='fs-6 text-muted'>Inténtalo de nuevo.</div>
    `;
  } else {
    // ESCENARIO 3: GAME OVER
    // Se agotan los intentos. Se revela la solución para feedback del usuario.
    inputUsuario.disabled = true;
    btnVerificar.disabled = true;
    
    resultadoDiv.innerHTML = `
        <span class='text-danger'>⛔ Se acabaron los intentos</span>
        <div class='fs-4 text-primary mt-2'>La solución era: <b>${solucionCorrecta}</b></div>
        <div class='fs-6 text-muted'>Pulsa "Generar nuevo" para reiniciar el ciclo.</div>
    `;
  }
}

// --- UTILIDADES DE UI ---

/**
 * Actualiza la interfaz gráfica del contador de intentos.
 * Aplica clases de Bootstrap dinámicamente para feedback semántico (Verde -> Amarillo -> Rojo).
 */
function actualizarBadgeIntentos() {
    const badge = document.getElementById('badge-intentos');
    badge.innerText = `Intentos restantes: ${intentosRestantes}`;
    
    // Limpieza y asignación de clases base.
    badge.className = 'badge mb-2 fs-6 '; 
    
    // Lógica de semáforo para el feedback visual.
    if (intentosRestantes === 3) {
        badge.classList.add('bg-success'); // Estado óptimo
    } else if (intentosRestantes === 2) {
        badge.classList.add('bg-warning', 'text-dark'); // Advertencia
    } else if (intentosRestantes === 1) {
        badge.classList.add('bg-danger'); // Estado crítico
    } else {
        badge.classList.add('bg-secondary'); // Estado inactivo
    }
}

inicializar();