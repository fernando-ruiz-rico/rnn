// Variables globales
let solucionCorrecta = []; // Ahora es un Array, no un string
let intentosRestantes = 5;
const MAX_INTENTOS = 5;

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

  // --- RESETER ESTADO ---
  intentosRestantes = MAX_INTENTOS;
  inputUsuario.value = "";
  inputUsuario.disabled = false;
  btnVerificar.disabled = false;
  inputUsuario.focus();
  resultadoDiv.innerHTML = "";
  
  actualizarBadgeIntentos();

  // UI Carga
  imgElement.style.display = 'none';
  spinner.style.display = 'block';

  try {
    // LLAMADA AL NUEVO ENDPOINT V2
    let respuesta = await fetch(`${URL_PYTHON}/captcha_v2`); 
    let data = await respuesta.json();

    if (data.error) throw new Error(data.error);

    // Nota: Las claves JSON deben coincidir con el backend (imagen_base64, solucion_real)
    imgElement.src = "data:image/png;base64," + data.img;
    solucionCorrecta = data.solucion; // Esperamos un Array: ["Camiseta", "Bota", ...]
    
    // Debug (Ocultar en producción real)
    console.log("Solución (Debug):", solucionCorrecta);

    imgElement.onload = () => {
        spinner.style.display = 'none';
        imgElement.style.display = 'block';
    };

  } catch (error) {
    console.error("Error:", error);
    spinner.style.display = 'none';
    resultadoDiv.innerHTML = "<span class='text-danger'>Error de conexión con el Generador de Moda</span>";
  }
}

// Lógica de verificación adaptada para listas de palabras
function verificarCaptcha(intento) {
  const resultadoDiv = document.getElementById('resultado');
  const inputUsuario = document.getElementById('intento-usuario');
  const btnVerificar = document.getElementById('btn-verificar');

  if (!intento || intento.trim() === "") return;

  // 1. Normalizar entrada del usuario
  // Convertimos a minúsculas, separamos por espacios o comas y filtramos vacíos
  const palabrasUsuario = intento.toLowerCase()
    .split(/[\s,]+/) // Separa por espacio O coma
    .filter(p => p.length > 0);

  // 2. Normalizar solución correcta
  const palabrasSolucion = solucionCorrecta.map(s => s.toLowerCase());

  // 3. Comparar arrays
  let esCorrecto = true;
  if (palabrasUsuario.length !== palabrasSolucion.length) {
      esCorrecto = false;
  } else {
      for (let i = 0; i < palabrasSolucion.length; i++) {
          if (palabrasUsuario[i] !== palabrasSolucion[i]) {
              esCorrecto = false;
              break;
          }
      }
  }

  // --- CASO 1: ACIERTO ---
  if (esCorrecto) {
    resultadoDiv.innerHTML = `
        <span class='text-success'>✅ ¡CORRECTO!</span>
        <div class='fs-6 text-muted'>Has identificado las prendas.</div>
    `;
    inputUsuario.disabled = true;
    btnVerificar.disabled = true;
    return;
  } 

  // --- CASO 2: FALLO ---
  intentosRestantes--;
  actualizarBadgeIntentos();

  if (intentosRestantes > 0) {
    resultadoDiv.innerHTML = `
        <span class='text-warning fs-4'>❌ Incorrecto</span>
        <div class='fs-6 text-muted'>Revisa el orden y la ortografía.</div>
    `;
  } else {
    // --- CASO 3: GAME OVER ---
    inputUsuario.disabled = true;
    btnVerificar.disabled = true;
    
    // Mostrar la solución de forma legible
    const solucionTexto = solucionCorrecta.join(", ");
    
    resultadoDiv.innerHTML = `
        <span class='text-danger'>⛔ Fin del juego</span>
        <div class='fs-5 text-primary mt-2'>Eran: <b>${solucionTexto}</b></div>
        <div class='fs-6 text-muted'>Pulsa "Generar nuevo" para reintentar.</div>
    `;
  }
}

function actualizarBadgeIntentos() {
    const badge = document.getElementById('badge-intentos');
    badge.innerText = `Intentos restantes: ${intentosRestantes}`;
    
    badge.className = 'badge mb-2 fs-6 ';
    
    if (intentosRestantes === 3) badge.classList.add('bg-success');
    else if (intentosRestantes === 2) badge.classList.add('bg-warning', 'text-dark');
    else if (intentosRestantes === 1) badge.classList.add('bg-danger');
    else badge.classList.add('bg-secondary');
}

inicializar();