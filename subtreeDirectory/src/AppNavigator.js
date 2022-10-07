import React, { useContext, useEffect, useState } from 'react'
import HomeScreen from './screens/HomeScreen';
import SigninScreen from './screens/SigninScreen';
import DashScreen from './screens/DashScreen';
import { AuthProvider } from './contexts/AuthContext';
import { BrowserRouter, Route, Routes } from 'react-router-dom'

export const AppNavigator = () => {
    // const [ isLoading, setIsLoading] = useState(true);

    return (
        <AuthProvider>
            <BrowserRouter>
                <Routes>
                    <Route index element={<HomeScreen />} />
                    <Route path="/signin" element={<SigninScreen />} />
                    <Route path="/dashboard" element={<DashScreen />} />
                    <Route path="*" element={<HomeScreen />} />
                </Routes>
            </BrowserRouter>
        </AuthProvider>
    )
}