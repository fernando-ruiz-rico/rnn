const FILAS = 6;
const COLUMNAS = 7;
// Representación lógica del estado del juego: Matriz de 6x7.
let tablero = []; 
let juegoActivo = true;
// Flag para evitar entradas del usuario mientras la IA "piensa" o se actualiza la UI.
let turnoBloqueado = false;

/**
 * Inicializa el estado lógico y visual del tablero.
 * Se encarga de limpiar el contenedor y regenerar la estructura DOM.
 */
function inicializarTablero() {
    const contenedor = document.getElementById('tablero');
    contenedor.innerHTML = '';
    tablero = [];

    for (let r = 0; r < FILAS; r++) {
        let filaLogica = [];
        // Contenedor visual para la fila (Bootstrap flex)
        let filaVisual = document.createElement('div');
        filaVisual.className = 'd-flex justify-content-center';
        
        for (let c = 0; c < COLUMNAS; c++) {
            filaLogica.push(0); // 0 representa casilla vacía
            
            // Creación de la celda visual
            let celda = document.createElement('div');
            celda.className = 'bg-light border border-2 border-dark rounded-circle m-1';
            celda.style.width = '60px';
            celda.style.height = '60px';
            celda.style.cursor = 'pointer';
            
            // Vinculación del evento de clic a la columna correspondiente
            celda.onclick = () => realizarJugadaHumano(c);
            // Identificador único para actualizaciones directas del DOM
            celda.id = `celda-${r}-${c}`;
            
            filaVisual.appendChild(celda);
        }
        tablero.push(filaLogica);
        contenedor.appendChild(filaVisual);
    }
    juegoActivo = true;
    turnoBloqueado = false;
    actualizarEstado("Tu turno (Rojo)", "alert-info");
}

/**
 * Sincroniza la matriz lógica `tablero` con la representación visual.
 * Se itera sobre todas las celdas para aplicar las clases de color correspondientes.
 */
function actualizarTableroVisual() {
    for (let r = 0; r < FILAS; r++) {
        for (let c = 0; c < COLUMNAS; c++) {
            const celda = document.getElementById(`celda-${r}-${c}`);
            celda.classList.remove('bg-light', 'bg-danger', 'bg-warning');
            
            if (tablero[r][c] === 0) {
                celda.classList.add('bg-light');
            } else if (tablero[r][c] === 1) {
                celda.classList.add('bg-danger'); // Jugador 1 (Humano)
            } else if (tablero[r][c] === 2) {
                celda.classList.add('bg-warning'); // Jugador 2 (IA)
            }
        }
    }
}

/**
 * Gestiona el flujo del turno del jugador humano.
 * Incluye validaciones de estado y llamadas asíncronas a la IA.
 */
async function realizarJugadaHumano(col) {
    if (!juegoActivo || turnoBloqueado) return;

    // Intenta colocar la ficha física
    if (colocarFicha(col, 1)) {
        actualizarTableroVisual();
        
        // Verificamos condición de victoria post-movimiento
        if (verificarFinJuego(1)) return;

        // Bloqueo de UI e inicio del turno de la IA
        turnoBloqueado = true;
        actualizarEstado("Pensando...", "alert-secondary");
        
        try {
            await realizarJugadaIA();
        } catch (error) {
            console.error(error);
            actualizarEstado("Error de conexión", "alert-dark");
            // Liberamos el turno en caso de fallo crítico para permitir reintento o reinicio
            turnoBloqueado = false;
        }
    }
}

/**
 * Lógica del agente inteligente.
 * Serializa el tablero y consulta al endpoint de Python para obtener el siguiente movimiento.
 */
async function realizarJugadaIA() {
    // Aplanamos la matriz para facilitar el transporte vía query param (formato CSV simple)
    const arrayPlano = tablero.flat();
    const tableroStr = arrayPlano.join(',');

    try {
        const respuesta = await fetch(`${URL_PYTHON}/conecta4?tablero=${encodeURIComponent(tableroStr)}`);
        const datos = await respuesta.json();

        if (datos.error) {
            console.error(datos.error);
            return;
        }

        const colIA = datos.columna;

        // Introducimos un delay artificial (400ms) para mejorar la UX (evita parpadeos instantáneos)
        setTimeout(() => {
            if (colocarFicha(colIA, 2)) {
                actualizarTableroVisual();
                // Verificación de victoria para la IA
                if (!verificarFinJuego(2)) {
                    actualizarEstado("Tu turno (Rojo)", "alert-info");
                    turnoBloqueado = false;
                }
            } else {
                // Fallback de seguridad: columna llena (edge case poco probable con modelo entrenado)
                turnoBloqueado = false; 
            }
        }, 400);

    } catch (e) {
        console.error("Error al contactar API:", e);
        turnoBloqueado = false;
    }
}

/**
 * Simula la gravedad del juego.
 * Recorre la columna desde abajo hacia arriba para encontrar el primer hueco disponible.
 */
function colocarFicha(col, jugador) {
    for (let r = FILAS - 1; r >= 0; r--) {
        if (tablero[r][col] === 0) {
            tablero[r][col] = jugador;
            return true;
        }
    }
    return false;
}

// --- LÓGICA DE VICTORIA ---

/**
 * Orquestador de fin de juego.
 * Comprueba victoria o empate y actualiza la UI acorde.
 */
function verificarFinJuego(jugador) {
    // 1. Comprobar victoria algorítmica
    if (comprobarVictoria(jugador)) {
        if (jugador === 1) {
            actualizarEstado("¡Felicidades! Has ganado 🎉", "alert-success");
        } else {
            actualizarEstado("La IA ha ganado 🤖", "alert-danger");
        }
        juegoActivo = false;
        return true;
    }

    // 2. Comprobar empate (tablero lleno sin ganador)
    if (tablero.flat().every(c => c !== 0)) {
        actualizarEstado("¡Empate!", "alert-dark");
        juegoActivo = false;
        return true;
    }

    return false;
}

/**
 * Algoritmo de comprobación de patrones ganadores (4 en raya).
 * Verifica direcciones: Horizontal, Vertical y ambas Diagonales.
 */
function comprobarVictoria(jugador) {
    // Horizontales
    for (let r = 0; r < FILAS; r++) {
        for (let c = 0; c < COLUMNAS - 3; c++) {
            if (tablero[r][c] === jugador && tablero[r][c+1] === jugador && 
                tablero[r][c+2] === jugador && tablero[r][c+3] === jugador) {
                return true;
            }
        }
    }
    // Verticales
    for (let r = 0; r < FILAS - 3; r++) {
        for (let c = 0; c < COLUMNAS; c++) {
            if (tablero[r][c] === jugador && tablero[r+1][c] === jugador && 
                tablero[r+2][c] === jugador && tablero[r+3][c] === jugador) {
                return true;
            }
        }
    }
    // Diagonales (hacia abajo-derecha)
    for (let r = 0; r < FILAS - 3; r++) {
        for (let c = 0; c < COLUMNAS - 3; c++) {
            if (tablero[r][c] === jugador && tablero[r+1][c+1] === jugador && 
                tablero[r+2][c+2] === jugador && tablero[r+3][c+3] === jugador) {
                return true;
            }
        }
    }
    // Diagonales (hacia arriba-derecha)
    for (let r = 3; r < FILAS; r++) {
        for (let c = 0; c < COLUMNAS - 3; c++) {
            if (tablero[r][c] === jugador && tablero[r-1][c+1] === jugador && 
                tablero[r-2][c+2] === jugador && tablero[r-3][c+3] === jugador) {
                return true;
            }
        }
    }
    return false;
}

function actualizarEstado(msj, clase) {
    const el = document.getElementById('status');
    el.innerText = msj;
    el.className = `alert ${clase} d-inline-block shadow-sm fw-bold`;
}

function reiniciarJuego() {
    inicializarTablero();
}

// Inicialización automática al cargar el script
inicializarTablero();