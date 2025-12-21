// Variables globales
// Guardamos la solución real que nos manda el back para comparar luego (ojo, en un entorno real muy seguro esto no debería estar tan expuesto, pero para esto sirve).
let solucionCorrecta = "";
// Control de vidas del usuario. Empezamos con 3.
let intentosRestantes = 3;
const MAX_INTENTOS = 3;

// Inicialización
// Aquí arrancamos todo: listeners de botones y primera carga.
function inicializar() {
  cargarNuevoCaptcha();

  // Listener para el botón de verificar
  document.getElementById('btn-verificar').addEventListener('click', function() {
    const intento = document.getElementById('intento-usuario').value;
    verificarCaptcha(intento);
  });

  // Botón para refrescar si no se ve bien la imagen o si ya has perdido
  document.getElementById('btn-recargar').addEventListener('click', function() {
    cargarNuevoCaptcha();
  });
  
  // Permitir verificar pulsando Enter, que es más cómodo para el usuario
  document.getElementById('intento-usuario').addEventListener('keypress', function(e) {
    if (e.key === 'Enter') {
      const intento = this.value;
      verificarCaptcha(intento);
    }
  });
}

// Carga imagen y resetea contadores
// Esta función es asíncrona porque tiene que pedirle la imagen al Python
async function cargarNuevoCaptcha() {
  const imgElement = document.getElementById('imagen-captcha');
  const spinner = document.getElementById('spinner-carga');
  const resultadoDiv = document.getElementById('resultado');
  const inputUsuario = document.getElementById('intento-usuario');
  const btnVerificar = document.getElementById('btn-verificar');
  const badgeIntentos = document.getElementById('badge-intentos');

  // --- RESETER ESTADO ---
  // Importante: cada vez que cargamos, reiniciamos las vidas y limpiamos inputs
  intentosRestantes = MAX_INTENTOS; // Volvemos a 3 intentos
  inputUsuario.value = "";
  inputUsuario.disabled = false;    // Habilitamos input por si estaba bloqueado de antes
  btnVerificar.disabled = false;    // Habilitamos botón
  inputUsuario.focus();             // Ponemos el foco ahí para escribir directo
  resultadoDiv.innerHTML = "";
  
  // Actualizar visual del contador (colores y texto)
  actualizarBadgeIntentos();

  // Interfaz de carga: ocultamos imagen vieja, mostramos spinner
  imgElement.style.display = 'none';
  spinner.style.display = 'block';

  try {
    // Pedimos el captcha al backend (Python/Flask)
    let respuesta = await fetch(`${URL_PYTHON}/captcha`);
    let data = await respuesta.json();

    if (data.error) throw new Error(data.error);

    // La imagen viene en base64, así que se la enchufamos directo al src
    imgElement.src = "data:image/png;base64," + data.img;
    solucionCorrecta = data.solucion;
    
    // Debug: Esto déjalo para desarrollo, pero quítalo en producción para que no hagan trampas mirando la consola ;)
    console.log("Solución (Debug):", solucionCorrecta);

    // Esperamos a que la imagen cargue de verdad para quitar el spinner
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

  // Validar que haya escrito algo, no gastemos intentos a lo tonto
  if (!intento || intento.trim() === "") {
     return; // No hacemos nada si está vacío
  }

  // --- CASO 1: ACIERTO ---
  // Comparamos lo que escribió con lo que guardamos antes
  if (intento.trim() === solucionCorrecta.trim()) {
    resultadoDiv.innerHTML = `
        <span class='text-success'>✅ ¡CORRECTO!</span>
        <div class='fs-6 text-muted'>Eres humano.</div>
    `;
    inputUsuario.disabled = true; // Bloqueamos para que no siga jugando si ya ganó
    btnVerificar.disabled = true;
    return;
  } 

  // --- CASO 2: FALLO ---
  // Restamos un intento
  intentosRestantes--;
  actualizarBadgeIntentos(); // Actualizamos el color y texto del badge

  if (intentosRestantes > 0) {
    // Todavía le quedan vidas
    inputUsuario.value = ""; // Borramos lo que escribió para que intente de nuevo rápido
    inputUsuario.focus();
    resultadoDiv.innerHTML = `
        <span class='text-warning fs-4'>❌ Incorrecto</span>
        <div class='fs-6 text-muted'>Inténtalo de nuevo.</div>
    `;
  } else {
    // --- CASO 3: GAME OVER (0 intentos) ---
    // Se acabó, bloqueamos todo y le chivamos la solución
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
// Esto es pura cosmética para dar feedback visual (verde -> amarillo -> rojo)
function actualizarBadgeIntentos() {
    const badge = document.getElementById('badge-intentos');
    badge.innerText = `Intentos restantes: ${intentosRestantes}`;
    
    // Cambiar colores dinámicamente reseteando clases
    badge.className = 'badge mb-2 fs-6 '; // Clases base
    
    if (intentosRestantes === 3) {
        badge.classList.add('bg-success'); // Verde
    } else if (intentosRestantes === 2) {
        badge.classList.add('bg-warning', 'text-dark'); // Amarillo (con texto oscuro para contraste)
    } else if (intentosRestantes === 1) {
        badge.classList.add('bg-danger'); // Rojo (peligro)
    } else {
        badge.classList.add('bg-secondary'); // Gris (0 intentos, muerto)
    }
}

inicializar();