import axios from 'axios';

const instance = axios.create({
    baseURL: 'https://localhost:3000'
})

instance.interceptors.request.use(
    async (config) => {
        return config
    },
    (err) => {
        return Promise.reject(err)
    }
)

export default instance;