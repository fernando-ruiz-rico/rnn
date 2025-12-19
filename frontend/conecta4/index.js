const FILAS = 6;
const COLUMNAS = 7;
let tablero = []; 
let juegoActivo = true;
let turnoBloqueado = false;

function inicializarTablero() {
    const contenedor = document.getElementById('tablero');
    contenedor.innerHTML = '';
    tablero = [];

    for (let r = 0; r < FILAS; r++) {
        let filaLogica = [];
        let filaVisual = document.createElement('div');
        filaVisual.className = 'd-flex justify-content-center';
        
        for (let c = 0; c < COLUMNAS; c++) {
            filaLogica.push(0);
            
            let celda = document.createElement('div');
            celda.className = 'bg-light border border-2 border-dark rounded-circle m-1';
            celda.style.width = '60px';
            celda.style.height = '60px';
            celda.style.cursor = 'pointer';
            
            celda.onclick = () => realizarJugadaHumano(c);
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

function actualizarTableroVisual() {
    for (let r = 0; r < FILAS; r++) {
        for (let c = 0; c < COLUMNAS; c++) {
            const celda = document.getElementById(`celda-${r}-${c}`);
            celda.classList.remove('bg-light', 'bg-danger', 'bg-warning');
            
            if (tablero[r][c] === 0) {
                celda.classList.add('bg-light');
            } else if (tablero[r][c] === 1) {
                celda.classList.add('bg-danger'); // Jugador
            } else if (tablero[r][c] === 2) {
                celda.classList.add('bg-warning'); // IA
            }
        }
    }
}

async function realizarJugadaHumano(col) {
    if (!juegoActivo || turnoBloqueado) return;

    if (colocarFicha(col, 1)) {
        actualizarTableroVisual();
        
        // Verificamos si ganó el humano (Jugador 1)
        if (verificarFinJuego(1)) return;

        turnoBloqueado = true;
        actualizarEstado("Pensando...", "alert-secondary");
        
        try {
            await realizarJugadaIA();
        } catch (error) {
            console.error(error);
            actualizarEstado("Error de conexión", "alert-dark");
            turnoBloqueado = false;
        }
    }
}

async function realizarJugadaIA() {
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

        setTimeout(() => {
            if (colocarFicha(colIA, 2)) {
                actualizarTableroVisual();
                // Verificamos si ganó la IA (Jugador 2)
                if (!verificarFinJuego(2)) {
                    actualizarEstado("Tu turno (Rojo)", "alert-info");
                    turnoBloqueado = false;
                }
            } else {
                // Si la columna está llena, intentamos mover en la primera libre (fallback simple)
                // Esto es raro que pase si la IA está bien entrenada
                turnoBloqueado = false; 
            }
        }, 400);

    } catch (e) {
        console.error("Error al contactar API:", e);
        turnoBloqueado = false;
    }
}

function colocarFicha(col, jugador) {
    for (let r = FILAS - 1; r >= 0; r--) {
        if (tablero[r][col] === 0) {
            tablero[r][col] = jugador;
            return true;
        }
    }
    return false;
}

// --- NUEVA LÓGICA DE VICTORIA ---
function verificarFinJuego(jugador) {
    // 1. Comprobar victoria
    if (comprobarVictoria(jugador)) {
        if (jugador === 1) {
            actualizarEstado("¡Felicidades! Has ganado 🎉", "alert-success");
        } else {
            actualizarEstado("La IA ha ganado 🤖", "alert-danger");
        }
        juegoActivo = false;
        return true;
    }

    // 2. Comprobar empate
    if (tablero.flat().every(c => c !== 0)) {
        actualizarEstado("¡Empate!", "alert-dark");
        juegoActivo = false;
        return true;
    }

    return false;
}

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

inicializarTablero();