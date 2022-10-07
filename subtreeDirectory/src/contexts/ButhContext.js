import createDataContext from './createDataContext'
import teleApi from '../apis/teleapi'

const authReducer = (state, action) => {
    switch (action.type) {
        case 'auth':
            return { ...state, errorMessage: '', token: action.payload, isLogged: true}
        case 'signout':
            return { ...state, token: '', isLogged: false}
        case 'fetch_data':
            return { ...state, data: action.payload}
        case 'clear_error_message':
            return { ...state, errorMessage: '', modalVisible: false}
        case 'add_error_message':
            return { ...state, errorMessage: action.payload}
        default:
            return state
    }
}

const clearErrorMessage = dispatch => () => {
    dispatch({ type: 'clear_error_message'})
}

const signin = dispatch => async ({ email, password }) => {
    try {
        const response = await teleApi.post('/users/login', { email, password })
        dispatch({ type: 'auth', payload: response.date.token})
    } catch (e) {
        dispatch({ type: 'add_error_message', payload: 'Something went wrong here'})
    }
}

const fetchData = dispatch => async ({ token }) => {
    try {
        const response = await teleApi.get('/users/me', {}, { 
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })
        dispatch({ type: 'fetch_data', payload: response.data})
    } catch (e) {
        dispatch({ type: 'add_error_message', payload: 'Cannot fetch, error'})
    }
}

const signout = dispatch => async () => {
    dispatch({ type: 'signout' })
}

export const { Provider, Context } = createDataContext(
    authReducer,
    { signin, signout, clearErrorMessage, fetchData },
    { 
        token: null,
        errorMessage: '', 
        isLogged: false,
        date: null
    }
)